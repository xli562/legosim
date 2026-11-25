
#include <fcntl.h>
#include <poll.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <signal.h>

#include <atomic>
#include <cerrno>
#include <ctime>
#include <vector>

#include "benchmark_yaml.h"
#include "cmd_handler.h"
#include "cmdline_options.h"
#include "spdlog/spdlog.h"

int flit_num = 2;  // Default flit size is 2*8 bytes
/**
 * @brief Data structure of process configuration.
 */
class ProcessStruct {
   public:
    ProcessStruct(const ProcessConfig& __config)
        : m_command(__config.m_command),
          m_args(__config.m_args),
          m_log_file(__config.m_log_file),
          m_to_stdout(__config.m_to_stdout),
          m_clock_rate(__config.m_clock_rate),
          m_pre_copy(__config.m_pre_copy),
          m_unfinished_line(),
          m_thread_id(),
          m_pid(-1),
          m_pid2(-1),
          m_sync_struct(NULL) {}

   public:
    // Configuration.
    std::string m_command;
    std::vector<std::string> m_args;
    std::string m_log_file;
    bool m_to_stdout;
    double m_clock_rate;
    std::string m_pre_copy;

    std::string m_unfinished_line;

    // Indentify
    int m_round;
    int m_phase;
    int m_thread;

    pthread_t m_thread_id;
    int m_pid;
    int m_pid2;
    bool m_succeeded = false;
    std::atomic<bool>* m_phase_failed = nullptr;

    /**
     * @brief Pointer to synchronize structure.
     */
    SyncStruct* m_sync_struct;
};

void parse_command(char* __pipe_buf, ProcessStruct* __proc_struct, int __stdin_fd) {
    // Split line by '\n'
    std::string line = std::string(__pipe_buf);
    std::vector<std::string> lines;

    int start_idx = 0;
    for (std::size_t i = 0; i < line.size(); i++) {
        if (line[i] == '\n') {
            std::string l = line.substr(start_idx, i + 1 - start_idx);
            start_idx = i + 1;
            lines.push_back(l);
        }
    }
    if (start_idx < line.size()) {
        std::string l = line.substr(start_idx, line.size() - start_idx);
        lines.push_back(l);
    }

    // Unfinished line.
    if (__proc_struct->m_unfinished_line.size() > 0) {
        lines[0] = __proc_struct->m_unfinished_line + lines[0];
        __proc_struct->m_unfinished_line = "";
    }
    if (lines[lines.size() - 1].find('\n') == -1) {
        __proc_struct->m_unfinished_line = lines[lines.size() - 1];
        lines.pop_back();
    }

    // Get line start with [INTERCMD]
    for (std::size_t i = 0; i < lines.size(); i++) {
        std::string l = lines[i];
        if (l.substr(0, 10) == "[INTERCMD]") {
            InterChiplet::SyncCommand cmd = InterChiplet::parseCmd(l);
            cmd.m_stdin_fd = __stdin_fd;
            cmd.m_clock_rate = __proc_struct->m_clock_rate;
            cmd.m_cycle = cmd.m_cycle / __proc_struct->m_clock_rate;

            pthread_mutex_lock(&__proc_struct->m_sync_struct->m_mutex);

            // Check order and clear delay infomation.
            if (cmd.m_type == InterChiplet::SC_BARRIER || cmd.m_type == InterChiplet::SC_SEND ||
                cmd.m_type == InterChiplet::SC_LOCK || cmd.m_type == InterChiplet::SC_UNLOCK ||
                cmd.m_type == InterChiplet::SC_LAUNCH) {
                InterChiplet::SyncCommand write_cmd = cmd;
                if (cmd.m_type == InterChiplet::SC_BARRIER) {
                    write_cmd.m_desc |= InterChiplet::SPD_BARRIER;
                    write_cmd.m_desc |= write_cmd.m_nbytes;
                } else if (cmd.m_type == InterChiplet::SC_LOCK) {
                    write_cmd.m_desc |= InterChiplet::SPD_LOCK;
                } else if (cmd.m_type == InterChiplet::SC_UNLOCK) {
                    write_cmd.m_desc |= InterChiplet::SPD_UNLOCK;
                } else if (cmd.m_type == InterChiplet::SC_LAUNCH) {
                    write_cmd.m_desc |= InterChiplet::SPD_LAUNCH;
                }
                if (!__proc_struct->m_sync_struct->m_delay_list.checkOrderOfCommand(write_cmd)) {
                    spdlog::warn("Command order has changed. Delay information is canncelled.");
                    __proc_struct->m_sync_struct->m_delay_list.clearDelayInfo();
                }
            }

            // Call functions to handle corresponding command.
            switch (cmd.m_type) {
                case InterChiplet::SC_CYCLE:
                    handle_cycle_cmd(cmd, __proc_struct->m_sync_struct);
                    break;
                case InterChiplet::SC_BARRIER:
                    handle_barrier_cmd(cmd, __proc_struct->m_sync_struct);
                    break;
                case InterChiplet::SC_SEND:
                case InterChiplet::SC_RECEIVE:
                    handle_pipe_cmd(cmd, __proc_struct->m_sync_struct);
                    break;
                case InterChiplet::SC_LOCK:
                    handle_lock_cmd(cmd, __proc_struct->m_sync_struct);
                    break;
                case InterChiplet::SC_UNLOCK:
                    handle_unlock_cmd(cmd, __proc_struct->m_sync_struct);
                    break;
                case InterChiplet::SC_LAUNCH:
                    handle_launch_cmd(cmd, __proc_struct->m_sync_struct);
                    break;
                case InterChiplet::SC_WAITLAUNCH:
                    handle_waitlaunch_cmd(cmd, __proc_struct->m_sync_struct);
                    break;
                case InterChiplet::SC_READ:
                    handle_read_cmd(cmd, __proc_struct->m_sync_struct, flit_num);
                    break;
                case InterChiplet::SC_WRITE:
                    handle_write_cmd(cmd, __proc_struct->m_sync_struct, flit_num);
                    break;
                default:
                    break;
            }

            pthread_mutex_unlock(&__proc_struct->m_sync_struct->m_mutex);
        }
    }
}

#define PIPE_BUF_SIZE 1024

void* bridge_thread(void* __args_ptr) {
    ProcessStruct* proc_struct = (ProcessStruct*)__args_ptr;

    int pipe_stdin[2];   // Pipe to send data to child process
    int pipe_stdout[2];  // Pipe to receive data from child process
    int pipe_stderr[2];  // Pipe to receive data from child process

    // Create pipes
    if (pipe(pipe_stdin) == -1 || pipe(pipe_stdout) == -1 || pipe(pipe_stderr) == -1) {
        perror("pipe");
        exit(EXIT_FAILURE);
    }

    // Create sub directory for subprocess.
    char* sub_dir_path = new char[128];
    sprintf(sub_dir_path, "./proc_r%d_p%d_t%d", proc_struct->m_round, proc_struct->m_phase,
            proc_struct->m_thread);
    if (access(sub_dir_path, F_OK) == -1) {
        mkdir(sub_dir_path, 0775);
    }

    // Fork a child process
    pid_t pid = fork();
    if (pid == -1) {
        perror("fork");
        exit(EXIT_FAILURE);
    }

    if (pid == 0) {  // Child process
        if (setpgid(0, 0) == -1) {
            perror("setpgid");
            _exit(127);
        }
        // Close unnecessary pipe ends
        close(pipe_stdin[1]);
        close(pipe_stdout[0]);
        close(pipe_stderr[0]);

        // Redirect stdin and stdout to pipes
        dup2(pipe_stdin[0], STDIN_FILENO);
        dup2(pipe_stdout[1], STDOUT_FILENO);
        dup2(pipe_stderr[1], STDERR_FILENO);

        // Change directory to sub-process.
        if (chdir(sub_dir_path) < 0) {
            perror("chdir");
        }
        // TODO: Copy necessary configuration file.
        if (!proc_struct->m_pre_copy.empty()) {
            std::string cp_cmd = std::string("cp ") + proc_struct->m_pre_copy + " .";
            if (system(cp_cmd.c_str()) != 0) {
                perror("system");
            }
        }

        std::cout << "CWD: " << get_current_dir_name() << std::endl;

        // Build arguments.
        int argc = proc_struct->m_args.size();
        char** args_list = new char*[argc + 2];
        args_list[0] = new char[proc_struct->m_command.size() + 1];
        strcpy(args_list[0], proc_struct->m_command.c_str());
        args_list[0][proc_struct->m_command.size()] = '\0';
        for (int i = 0; i < argc; i++) {
            int arg_len = proc_struct->m_args[i].size();
            args_list[i + 1] = new char[arg_len + 1];
            strcpy(args_list[i + 1], proc_struct->m_args[i].c_str());
            args_list[i + 1][arg_len] = '\0';
        }
        args_list[argc + 1] = NULL;

        // Execute the child program
        std::cout << "Exec: ";
        for (int i = 0; i < proc_struct->m_args.size() + 1; i++) {
            std::cout << " " << args_list[i];
        }
        std::cout << std::endl;
        execvp(args_list[0], args_list);

        // If execl fails, it means the child process couldn't be started
        perror("execvp");
        _exit(127);
    } else {  // Parent process
        // Both sides establish the group to avoid racing the child's exec.
        if (setpgid(pid, pid) == -1 && errno != EACCES && errno != ESRCH) {
            perror("setpgid");
            proc_struct->m_phase_failed->store(true);
        }
        spdlog::info("Start simulation process {}. Command: {}", pid, proc_struct->m_command);
        proc_struct->m_pid2 = pid;

        // Close unnecessary pipe ends
        close(pipe_stdin[0]);
        close(pipe_stdout[1]);
        close(pipe_stderr[1]);
        int stdin_fd = pipe_stdin[1];
        int stdout_fd = pipe_stdout[0];
        int stderr_fd = pipe_stderr[0];

        pollfd fd_list[2] = {{fd : stdout_fd, events : POLL_IN},
                             {fd : stderr_fd, events : POLL_IN}};

        // Move log to subfolder.
        std::ofstream log_file(std::string(sub_dir_path) + "/" + proc_struct->m_log_file);

        // Write execution start time to log file.
        std::time_t t = std::time(0);
        std::tm* now = std::localtime(&t);
        log_file << "Execution starts at " << (now->tm_year + 1900) << "-" << (now->tm_mon + 1)
                 << "-" << now->tm_mday << "  " << (now->tm_hour) << ":" << (now->tm_min) << ":"
                 << (now->tm_sec) << std::endl;

        char* pipe_buf = new char[PIPE_BUF_SIZE + 1];
        bool to_stdout = proc_struct->m_to_stdout;
        bool io_ok = true;
        int status = 0;
        pid_t waited = 0;
        while (true) {
            if (proc_struct->m_phase_failed->load()) {
                spdlog::warn("Stopping simulator {} (pid {}) after phase failure.",
                             proc_struct->m_command, pid);
                kill(-pid, SIGKILL);
                if (waited == 0) kill(pid, SIGKILL);
                io_ok = false;
                break;
            }
            int ready = poll(fd_list, 2, 1000);
            if (ready == -1) {
                if (errno == EINTR) continue;
                perror("poll");
                io_ok = false;
                break;
            }

            bool has_output = false;
            for (int stream = 0; stream < 2; ++stream) {
                if (fd_list[stream].revents & (POLLERR | POLLNVAL)) {
                    spdlog::error("Output pipe failed for process {}.", pid);
                    io_ok = false;
                    break;
                }
                if (!(fd_list[stream].revents & (POLLIN | POLLHUP))) continue;
                ssize_t count = read(fd_list[stream].fd, pipe_buf, PIPE_BUF_SIZE);
                if (count < 0) {
                    if (errno == EINTR) continue;
                    perror("read");
                    io_ok = false;
                    break;
                }
                if (count == 0) {
                    // Disable EOF streams while continuing to monitor the child.
                    fd_list[stream].fd = -1;
                    continue;
                }
                has_output = true;
                pipe_buf[count] = '\0';
                log_file.write(pipe_buf, count).flush();
                if (to_stdout) {
                    std::ostream& output = stream == 0 ? std::cout : std::cerr;
                    output.write(pipe_buf, count).flush();
                }
                parse_command(pipe_buf, proc_struct, stdin_fd);
            }
            if (!io_ok) break;

            // Drain available output before reaping, including final log messages.
            if (waited == 0) {
                waited = waitpid(pid, &status, WNOHANG);
                if (waited == -1 && errno == EINTR) {
                    waited = 0;
                    continue;
                }
                if (waited == -1 || (waited > 0 &&
                    (!WIFEXITED(status) || WEXITSTATUS(status) != 0))) {
                    proc_struct->m_phase_failed->store(true);
                    kill(-pid, SIGKILL);
                    break;
                }
            }
            if (waited != 0 && !has_output) break;
        }

        if (!io_ok) {
            proc_struct->m_phase_failed->store(true);
            kill(-pid, SIGKILL);
            if (waited == 0) kill(pid, SIGKILL);
        }
        close(stdin_fd);
        close(stdout_fd);
        close(stderr_fd);
        if (waited == 0) {
            do {
                waited = waitpid(pid, &status, 0);
            } while (waited == -1 && errno == EINTR);
        }
        if (waited == -1) {
            spdlog::error("waitpid failed for {} (pid {}): errno {}",
                          proc_struct->m_command, pid, errno);
        } else if (WIFSIGNALED(status)) {
            spdlog::error("{} (pid {}) terminated by signal {}.",
                          proc_struct->m_command, pid, WTERMSIG(status));
        } else if (WIFEXITED(status)) {
            int exit_code = WEXITSTATUS(status);
            proc_struct->m_succeeded = io_ok && exit_code == 0;
            if (exit_code == 0) {
                spdlog::info("Simulation process {} exited with status 0.", pid);
            } else {
                spdlog::error("{} (pid {}) exited with status {}.",
                              proc_struct->m_command, pid, exit_code);
            }
        }
        delete[] pipe_buf;
    }
    delete[] sub_dir_path;

    return 0;
}

struct PhaseOneResult {
    InterChiplet::InnerTimeType cycles;
    bool succeeded;
};

PhaseOneResult __loop_phase_one(
    int __round, const std::vector<ProcessConfig>& __proc_phase1_cfg_list,
    const std::vector<ProcessConfig>& __proc_phase2_cfg_list, int __width) {
    // Create synchronize data structure.
    SyncStruct* g_sync_structure = new SyncStruct();
    std::atomic<bool> phase_failed(false);

    // Load delay record.
    g_sync_structure->m_delay_list.loadDelayInfo("delayInfo.txt",
                                                 __proc_phase2_cfg_list[0].m_clock_rate, __width);
    spdlog::info("Load {} delay records.", g_sync_structure->m_delay_list.size());

    // Create multi-thread.
    int thread_i = 0;
    std::vector<ProcessStruct*> proc_struct_list;
    for (auto& proc_cfg : __proc_phase1_cfg_list) {
        ProcessStruct* proc_struct = new ProcessStruct(proc_cfg);
        proc_struct->m_round = __round;
        proc_struct->m_phase = 1;
        proc_struct->m_thread = thread_i;
        proc_struct->m_sync_struct = g_sync_structure;
        proc_struct->m_phase_failed = &phase_failed;
        int res =
            pthread_create(&(proc_struct->m_thread_id), NULL, bridge_thread, (void*)proc_struct);
        if (res != 0) {
            spdlog::error("pthread_create failed: error {}", res);
            exit(EXIT_FAILURE);
        }

        proc_struct_list.push_back(proc_struct);
        thread_i++;
    }

    // Read worker results only after joining, so no synchronization is needed here.
    bool succeeded = true;
    for (auto& proc_struct : proc_struct_list) {
        int join_error = pthread_join(proc_struct->m_thread_id, NULL);
        if (join_error != 0) {
            spdlog::error("pthread_join failed: error {}", join_error);
            exit(EXIT_FAILURE);
        }
        succeeded = succeeded && proc_struct->m_succeeded;
        delete proc_struct;
    }
    spdlog::info("All processes have exited.");

    // Remove file.
    for (auto& pipe_name : g_sync_structure->m_pipe_struct.pipeSet()) {
        remove(pipe_name.c_str());
    }

    // Dump benchmark record.
    g_sync_structure->m_bench_list.dumpBench("bench.txt", __proc_phase2_cfg_list[0].m_clock_rate, __width);
    spdlog::info("Dump {} bench records.", g_sync_structure->m_bench_list.size());

    // Destory global synchronize structure, and return total cycle.
    InterChiplet::InnerTimeType res_cycle = g_sync_structure->m_cycle_struct.cycle();
    delete g_sync_structure;
    return {res_cycle, succeeded};
}

bool __loop_phase_two(int __round, const std::vector<ProcessConfig>& __proc_cfg_list) {
    // Create synchronize data structure.
    SyncStruct* g_sync_structure = new SyncStruct();
    std::atomic<bool> phase_failed(false);

    // Create multi-thread.
    int thread_i = 0;
    std::vector<ProcessStruct*> proc_struct_list;
    for (auto& proc_cfg : __proc_cfg_list) {
        ProcessStruct* proc_struct = new ProcessStruct(proc_cfg);
        proc_struct->m_round = __round;
        proc_struct->m_phase = 2;
        proc_struct->m_thread = thread_i;
        thread_i++;
        proc_struct->m_sync_struct = g_sync_structure;
        proc_struct->m_phase_failed = &phase_failed;
        int res =
            pthread_create(&(proc_struct->m_thread_id), NULL, bridge_thread, (void*)proc_struct);
        if (res != 0) {
            spdlog::error("pthread_create failed: error {}", res);
            exit(EXIT_FAILURE);
        }

        proc_struct_list.push_back(proc_struct);
        thread_i++;
    }

    // Read worker results only after joining, so no synchronization is needed here.
    bool succeeded = true;
    for (auto& proc_struct : proc_struct_list) {
        int join_error = pthread_join(proc_struct->m_thread_id, NULL);
        if (join_error != 0) {
            spdlog::error("pthread_join failed: error {}", join_error);
            exit(EXIT_FAILURE);
        }
        succeeded = succeeded && proc_struct->m_succeeded;
        delete proc_struct;
    }
    spdlog::info("All processes have exited.");

    // Destory global synchronize structure.
    delete g_sync_structure;
    return succeeded;
}

int main(int argc, const char* argv[]) {
    // A cancelled simulator can close its command pipe before a reply is sent.
    signal(SIGPIPE, SIG_IGN);
    // Parse command line.
    CmdLineOptions options;
    if (options.parse(argc, argv) != 0) {
        return 0;
    };
    int width = options.m_width;
    flit_num = options.m_flit_num;
    // Initializate logging
    if (options.m_debug) {
        spdlog::set_level(spdlog::level::debug);
    }
    spdlog::info("==== LegoSim Chiplet Simulator ====");

    // Change working directory if --cwd is specified.
    if (options.m_cwd != "") {
        if (access(options.m_cwd.c_str(), F_OK) == 0) {
            if (chdir(options.m_cwd.c_str()) < 0) {
                perror("chdir");
            }
            spdlog::info("Change working directory {}.", get_current_dir_name());
        }
    }

    // Check exist of benchmark configuration yaml.
    if (access(options.m_bench.c_str(), F_OK) < 0) {
        spdlog::error("Cannot find benchmark {}.", options.m_bench);
        exit(EXIT_FAILURE);
    }

    // Load benchmark configuration.
    BenchmarkConfig configs(options.m_bench);
    spdlog::info("Load benchmark configuration from {}.", options.m_bench);

    // Get start time of simulation.
    struct timeval simstart, simend, roundstart, roundend;
    gettimeofday(&simstart, 0);

    InterChiplet::InnerTimeType sim_cycle = 0;
    for (int round = 1; round <= options.m_timeout_threshold; round++) {
        // Get start time of one round.
        gettimeofday(&roundstart, 0);
        spdlog::info("**** Round {} Phase 1 ****", round);
        PhaseOneResult phase_one =
            __loop_phase_one(round, configs.m_phase1_proc_cfg_list, configs.m_phase2_proc_cfg_list,
                             width);
        if (!phase_one.succeeded) {
            spdlog::error("Round {} phase 1 failed.", round);
            return EXIT_FAILURE;
        }
        InterChiplet::InnerTimeType round_cycle = phase_one.cycles;

        // Get simulation cycle.
        // If simulation cycle this round is close to the previous one, quit iteration.
        spdlog::info("Benchmark elapses {} cycle.",
                     static_cast<InterChiplet::TimeType>(round_cycle));
        if (round > 1) {
            // Calculate error of simulation cycle.
            double err_rate = ((double)round_cycle - (double)sim_cycle) / (double)round_cycle;
            err_rate = err_rate < 0 ? -err_rate : err_rate;
            spdlog::info("Difference related to pervious round is {}%.", err_rate * 100);
            // If difference is small enough, quit loop.
            if (err_rate < options.m_err_rate_threshold) {
                spdlog::info("Quit simulation because simulation cycle has converged.");
                sim_cycle = round_cycle;
                break;
            }
        }
        sim_cycle = round_cycle;

        spdlog::info("**** Round {} Phase 2 ***", round);
        if (!__loop_phase_two(round, configs.m_phase2_proc_cfg_list)) {
            spdlog::error("Round {} phase 2 failed.", round);
            return EXIT_FAILURE;
        }

        // Get end time of one round.
        gettimeofday(&roundend, 0);
        unsigned long elaped_sec = roundend.tv_sec - roundstart.tv_sec;
        spdlog::info("Round {} elapses {}d {}h {}m {}s.", round, elaped_sec / 3600 / 24,
                     (elaped_sec / 3600) % 24, (elaped_sec / 60) % 60, elaped_sec % 60);
    }

    // Get end time of simulation.
    spdlog::info("**** End of Simulation ****");
    gettimeofday(&simend, 0);
    unsigned long elaped_sec = simend.tv_sec - simstart.tv_sec;
    spdlog::info("Benchmark elapses {} cycle.", static_cast<InterChiplet::TimeType>(sim_cycle));
    spdlog::info("Simulation elapsed {}d {}h {}m {}s.", elaped_sec / 3600 / 24,
                 (elaped_sec / 3600) % 24, (elaped_sec / 60) % 60, elaped_sec % 60);
}

// Small standalone front end for the DSENT source vendored under gem5/ext/dsent.
// The vendored CMake target builds a Python 2 extension, not an executable.
#include "DSENT.h"

#include <exception>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cerr << "usage: dsent_cli CONFIG.cfg\n";
        return 2;
    }

    std::map<LibUtil::String, LibUtil::String> config;
    DSENT::Model *model = NULL;
    try {
        model = DSENT::initialize(argv[1], config);
        std::map<std::string, double> values;
        DSENT::run(config, model, values);
        std::cout << std::setprecision(17);
        for (const auto &entry : values) {
            std::cout << "DSENT_RESULT " << entry.first << " " << entry.second << '\n';
        }
        DSENT::finalize(config, model);
    } catch (const std::exception &error) {
        std::cerr << "DSENT error: " << error.what() << '\n';
        if (model != NULL) DSENT::finalize(config, model);
        return 1;
    }
    return 0;
}

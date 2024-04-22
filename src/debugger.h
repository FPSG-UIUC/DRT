#ifndef DEBUGGER_H
#define DEBUGGER_H

#include <stdlib.h>
#include <string>
#include <iostream>

//Debugger class
//October 2020
//Inspired by: https://stackoverflow.com/questions/6406307/how-do-i-set-a-debug-mode-in-c

//Debug level -- each print message has a debug level
//Debug levels 0-2
//The debugger only prints messages below the level it is set to
//level 0 --> no messages
//level 1 --> low verbosity
//level 2 --> high verbosity
//If the debugger is set to 0, messages set to verbosity of 1 or above will not be printed.
class Debugger{
    private: 
        int debug_level;

    public:
        //Turn the debugger OFF by default
        Debugger() { 
            this->debug_level = 0; 
        }

        ~Debugger();

        void setDebug(int debug_level) {
            this->debug_level = debug_level;
        }


        void print(const char* message, int verbosity) {
            if (this->debug_level >= verbosity) {
                std::cout << "DEBUG " << std::to_string(verbosity) << ": " << message << "\n" << std::endl;
            }
        }
};

#endif

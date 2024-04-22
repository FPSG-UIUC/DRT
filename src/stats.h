#ifndef STATS_H
#define STATS_H

#include <stdint.h>

class Stats{
	public:
		Stats(){
			this->a_size_llb = 0;	this->b_size_llb = 0;	this->o_size_llb = 0;
			this->a_read_llb = 0;	this->b_read_llb = 0;	this->o_read_llb = 0;
			this->o_write_llb= 0; this->cleanRdWr_llb =0; this->runtime = 0.0;
			this->cycles= 0; this->pe_busy_cycles = 0;
			// Middle DOT NoC stats
			this->a_read_middle = 0;	this->b_read_middle = 0;	this->o_read_middle = 0;
			this->o_write_middle= 0;
			return;
		}

		// Get functions

		// The next three are not used anymore
		int Get_a_size(){return a_size_llb;}
		int Get_b_size(){return b_size_llb;}
		int Get_o_size(){return o_size_llb;}

		// Middle DOT Stats
		uint64_t Get_a_read_middle(){return a_read_middle;}
		uint64_t Get_b_read_middle(){return b_read_middle;}
		uint64_t Get_o_read_middle(){return o_read_middle;}
		uint64_t Get_o_write_middle(){return o_write_middle;}

		// Top DOT Stats
		uint64_t Get_a_read(){return a_read_llb;}
		uint64_t Get_b_read(){return b_read_llb;}
		uint64_t Get_o_read(){return o_read_llb;}
		uint64_t Get_o_write(){return o_write_llb;}

		double Get_runtime(){return this->runtime;}
		uint64_t Get_cycles(){return this->cycles;}
		uint64_t Get_pe_busy_cycles(){return this->pe_busy_cycles;}

		// Not used anymore
		int Get_llb_used_size(){return a_size_llb+b_size_llb+o_size_llb;}

		uint64_t Get_total_traffic(){return a_read_llb+b_read_llb+o_read_llb+o_write_llb;}
		uint64_t Get_total_traffic_middle(){return a_read_middle+b_read_middle+o_read_middle+o_write_middle;}

		uint64_t Get_output_traffic(){return o_read_llb+o_write_llb;}

		uint64_t Get_cleanRdWr(){return 2*cleanRdWr_llb;}

		// Set functions
		void Set_a_size(int a_size){ a_size_llb = a_size; return;}
		void Set_b_size(int b_size){ b_size_llb = b_size; return;}
		void Set_o_size(int o_size){ o_size_llb = o_size; return;}

		void Set_a_read(uint64_t a_read){ a_read_llb = a_read; return;}
		void Set_b_read(uint64_t b_read){ b_read_llb = b_read; return;}
		void Set_o_read(uint64_t o_read){ o_read_llb = o_read; return;}
		void Set_o_write(uint64_t o_write){ o_write_llb = o_write; return;}

		void Set_runtime(double runtime){this->runtime = runtime; return;}
		void Set_cycles(uint64_t cycles){this->cycles = cycles; return;}
		void Set_pe_busy_cycles(uint64_t pe_busy_cycles)
			{this->pe_busy_cycles = pe_busy_cycles; return;}

		// Accumulate functions
		// The next three are not used anymore
		void Accumulate_a_size(int a_size){ a_size_llb += a_size; return;}
		void Accumulate_b_size(int b_size){ b_size_llb += b_size; return;}
		void Accumulate_o_size(int o_size){ o_size_llb += o_size; return;}

		// Middle DOT stats
		void Accumulate_a_read_middle(uint64_t a_read_middle){ this->a_read_middle += a_read_middle; return;}
		void Accumulate_b_read_middle(uint64_t b_read_middle){ this->b_read_middle += b_read_middle; return;}
		void Accumulate_o_read_middle(uint64_t o_read_middle){ this->o_read_middle += o_read_middle; return;}
		void Accumulate_o_write_middle(uint64_t o_write_middle){ this->o_write_middle += o_write_middle; return;}

		// Top DOT
		void Accumulate_a_read(uint64_t a_read){ a_read_llb += a_read; return;}
		void Accumulate_b_read(uint64_t b_read){ this->b_read_llb += b_read; return;}
		void Accumulate_o_read(uint64_t o_read){ o_read_llb += o_read; return;}
		void Accumulate_o_write(uint64_t o_write){ o_write_llb += o_write; return;}
		void Accumulate_runtime(double runtime){this->runtime += runtime; return;}
		void Accumulate_cycles(uint64_t cycles){this->cycles += cycles; return;}
		void Accumulate_pe_busy_cycles(uint64_t pe_busy_cycles)
			{this->pe_busy_cycles += pe_busy_cycles; return;}
		void Accumulate_cleanRdWr(uint64_t cleanRdWr_llb)
			{this->cleanRdWr_llb += cleanRdWr_llb; return;}

		// Reset everything
		void Reset(){
			this->a_size_llb = 0;	this->b_size_llb = 0;	this->o_size_llb = 0;
			this->a_read_llb = 0;	this->b_read_llb = 0;	this->o_read_llb = 0;
			this->o_write_llb= 0; this->cleanRdWr_llb =0; this->runtime = 0.0;
			this->cycles= 0; this->pe_busy_cycles = 0;
			return;
		}
	private:
		// Size of A, B, and O in LLB memory
		// Legacy vals (next three); not used anymore
		int a_size_llb;
		int b_size_llb;
		int o_size_llb;
		// Bandwidth used for A, B, and O in Middle DOT NoC
		uint64_t a_read_middle;
		uint64_t b_read_middle;
		uint64_t o_read_middle;
		uint64_t o_write_middle;
		// Bandwidth used for A, B, and O in Top DOT NoC
		uint64_t a_read_llb;
		uint64_t b_read_llb;
		uint64_t o_read_llb;
		uint64_t o_write_llb;
		// Traffic spent for clean Rd and Wr
		uint64_t cleanRdWr_llb;
		// runtime
		double runtime;
		// Cycles
		uint64_t cycles;
		uint64_t pe_busy_cycles;

};
#endif

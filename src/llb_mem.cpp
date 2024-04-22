#ifndef LLB_MEM_C_
#define LLB_MEM_C_

#include "llb_mem.h"

// Add a PE tile(s) to LLB
// Updates the read/write statistics in case update_traffic is on
int LLB_Mem::AddToLLB(char mat_name, Req rd_wr, uint64_t bytes, int update_traffic){
	// There is no space left; early termination
	switch (mat_name){
		case 'A':
			a_size += bytes;
			if(update_traffic) stats->Accumulate_a_read(bytes);
			break;
		case 'B':
			b_size += bytes;
			if(update_traffic) stats->Accumulate_b_read(bytes);
			break;
		case 'O':
			o_size += bytes;
			if(update_traffic){
				if (rd_wr == Req::read)
					stats->Accumulate_o_read(bytes);
				else
					stats->Accumulate_o_write(bytes);
			}
			break;
		default: printf("Unknown variable is requested!\n"); exit(1);
	}
	// Update total used capacity bytes of the LLB
	used_size += bytes;

	//if(used_size > total_size)
	//	printf("Exceeding The Size! %f\n",(double)total_size/(1024.0*1024.0));

	// Log the maximum ever used LLB capacity
	if (used_size > max_used_size)
		max_used_size = used_size;

	return 1;
}

// Checks if a specific size of data can fit into LLB
// The routine will not take any action
// returns 0 upon a failure and 1 otherwise
int LLB_Mem::DoesFitInLLB(char mat_name, uint64_t bytes){
	// There is no space left; early termination
	if( (bytes+used_size) > total_size)
		return 0;
	// The switch case statement is for TACTile SpMSpM only
	switch (mat_name){
		case 'A':
			if((params->getCompKernel() == kernel::SpMSpM)
					& (params->getTilingMechanism() == tiling::t_dynamic) ){
				if((a_size+bytes) > max_a_size )
					return 0;
			}
			break;
		case 'B':
			if(params->getTilingMechanism() == tiling::t_dynamic){
				if((b_size+bytes) > max_b_size )
					return 0;
			}
			break;
		case 'O':
			// FIXME: This works for Scheduler_8! But please do not try it for other ones
			//	can break stuff :(
			// For scheduler_8 SpMSpM comment out the if statement to let output grow to maximum
			if((params->getCompKernel() == kernel::SpMSpM)
					& (params->getTilingMechanism() == tiling::t_dynamic) ){
				if((o_size+bytes) > max_o_size )
					return 0;
			}
			break;
		default: printf("Unknown variable is requested!\n"); exit(1);
	}
	return 1;
}

// Remove #bytes from the LLB portion of a specific tensor
// Updates the write statistics for O in case update_traffic is on
void LLB_Mem::RemoveFromLLB(char mat_name, Req rd_wr, uint64_t bytes, int update_traffic){
	// Sanity check
	if(bytes > used_size){
		printf("The LLB memory size is becoming negative!\n");
		exit(1);
	}
	switch (mat_name){
		case 'A':
			if(a_size < bytes){printf("A size is negative %lu, %lu\n", a_size, bytes); exit(1);}
			a_size -= bytes;
			break;
		case 'B':
			if(b_size < bytes){printf("B size is negative %lu, %lu\n", b_size, bytes); exit(1);}
			b_size -= bytes;
			break;
		case 'O':
			if(o_size < bytes){printf("O size is negative %lu, %lu\n", o_size, bytes); exit(1);}
			o_size -= bytes;
			if(update_traffic) stats->Accumulate_o_write(bytes);
			break;
		default: printf("Unknown variable is requested!\n"); exit(1);
	}
	used_size -= bytes;
	return;
}

// Completely erase the tiles corresponding to a matrix from LLB
// Updates the write statistics for O in case update_traffic is on
void LLB_Mem::EvictMatrixFromLLB(char mat_name, int update_traffic){
	switch (mat_name){
		case 'A': a_size = 0;
			break;
		case 'B': b_size = 0;
			break;
		case 'O':
				if(update_traffic) stats->Accumulate_o_write(o_size);
				o_size = 0;
			break;
		default: printf("Unknown variable is requested!\n"); exit(1);
	}
	this->used_size = a_size + b_size + o_size;
}


// Add data to LLB; This is a minimal version used for no-tiled versions
int LLB_Mem::AddToLLB_Minimal(uint64_t bytes){
	// Update total used capacity bytes of the LLB
	used_size += bytes;
	// Log the maximum ever used LLB capacity
	if (used_size > max_used_size)
		max_used_size = used_size;

	return 1;
}

// Remove #bytes from LLB; This is a minimal version used for no-tiled versions
void LLB_Mem::RemoveFromLLB_Minimal(uint64_t bytes){
	// Update the LLB used size
	used_size -= bytes;
	return;
}

// Checks if a specific size of data can fit into LLB
// The routine will not take any action
// returns 0 upon a failure and 1 otherwise
//  ; This is a minimal version used for no-tiled versions
int LLB_Mem::DoesFitInLLB_Minimal(uint64_t bytes){
	// There is no space left; early termination
	if( (bytes+used_size) > total_size)
		return 0;
	else
		return 1;
}


#endif

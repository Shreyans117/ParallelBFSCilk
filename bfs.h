#include <cmath>
#include <atomic>

using namespace std;

int* next_frontier_sparse(bool* visited, int* offset, int* E, int* dist, int* frontier, int &frontierLength, atomic<bool>* visited2);
int* filter(int &no_nghs_visited, int* offset, int* E, int* psflags, int start, int end, int node);
void scan_exclusive(int* ps, int n, size_t no_of_chunks);
int* flatten(int** nghs, int* frontierOffsets, int &oldFrontierLength);
size_t reduce(size_t* A, int n);
int* transformSparseToDense(bool* visited, int n, int* frontier, int frontierLength);
int* next_frontier_dense(bool* visited, int n, int* offset, int* E, int* dist, int* frontier, int &frontierLength);
int* transformtDenseToSparse(int n, int* frontier, int &frontierLength);

size_t reduce(int* A, int n) {
	int threshold = 7501;
	if (n<threshold) {
		size_t sum = 0;
		for (int i = 0; i < n; i++) {
			sum += A[i];
		}	
		return sum;
	}
	size_t L, R;
	L = cilk_spawn reduce(A, n/2);
	R = reduce(A+n/2, n-n/2);
	cilk_sync;
	return L+R;
}

void BFS(int n, int m, int* offset, int* E, int s, int* dist) {
    atomic<bool>* visited2 = new atomic<bool>[n];
    bool* visited = new bool[n];
    cilk_for(int i = 0; i < n; i++) {
        dist[i] = -1;
        visited[i]=false;
    }
    visited[s]=true;
    dist[s] = 0;
    int frontierLength = 1;
    int* frontier = new int[frontierLength];
    frontier[0] = s;
    bool modeSwitch=false;
    string previousMode = "sparse";
    string currentMode = "sparse";
    while (frontierLength > 0) {
        if(frontierLength>n/14) {
            currentMode = "dense";
        }
        else{
            currentMode = "sparse";
        }
        if (previousMode != currentMode) {
            modeSwitch=true;
        }
        if (currentMode == "dense") {
            if(modeSwitch) {
                frontier = transformSparseToDense(visited, n, frontier, frontierLength);
                modeSwitch=false;
            }
            frontier = next_frontier_dense(visited, n, offset, E, dist, frontier, frontierLength);
        }
        else if (currentMode == "sparse") {
            if(modeSwitch) {
                frontier=transformtDenseToSparse(n, frontier, frontierLength);
                modeSwitch=false;
            }
            frontier = next_frontier_sparse(visited, offset, E, dist, frontier, frontierLength, visited2);
        }
        previousMode = currentMode;
    }
}
int* transformtDenseToSparse(int n, int* frontier, int &frontierLength) {
    scan_exclusive(frontier, n, ceil(sqrt(n)));
    frontierLength = frontier[n-1];

    int* next_frontier = new int[frontierLength];

    if (frontier[0] == 1) {
        next_frontier[0] = 0;
    }
    cilk_for (int i = 1; i < n; i++) {
        if (frontier[i-1] != frontier[i])
            {
                next_frontier[frontier[i] - 1] = i; 
            }
    }
    return next_frontier;
}

int* next_frontier_dense(bool* visited, int n, int* offset, int* E, int* dist, int* frontier, int &frontierLength) {
    int* frontier_flags = new int[n];
    int* length = new int[n];
    cilk_for(int i = 0; i < n; i++) {
        frontier_flags[i] = 0;
    }
    cilk_for(int i = 0; i < n; i++) {
        if (!visited[i]) {
            int ngh;
            for (int j = offset[i]; j < offset[i+1]; j++) {
                ngh = E[j];
                if (frontier[ngh]==1) {
                    dist[i] = dist[ngh] + 1;
                    frontier_flags[i] = 1;
                    visited[i]=true;
                    break;
                }
            }
        }
    }
    
    frontierLength = reduce(frontier_flags, n);
    return frontier_flags;
}
int* transformSparseToDense(bool* visited, int n, int* frontier, int frontierLength) {
    int* next_frontier = new int[n];
    cilk_for(int i = 0; i < n; i++) {
        if(visited[i]) {
            next_frontier[i] = 1;
        }
        else next_frontier[i] = 0;
    }

    return next_frontier;
}
int* next_frontier_sparse(bool* visited, int* offset, int* E, int* dist, int* frontier, int &frontierLength, atomic<bool>* visited2) {
   
    int** nghs = new int*[frontierLength];
    int* frontierOffsets = new int[frontierLength];
    cilk_for(int k = 0; k < frontierLength; k++) {
        int x = frontier[k];
        int cur_dis=dist[x];
        int start = offset[x];
        int end = offset[x+1];
        int range = end - start;
        int nghsVisitedLength = 0;
        int* psflags = new int[range];

        for(size_t l = 0; l < range; l++) psflags[l] = 0;

        if (range < 2001) {
            for(int j = start; j < end; j++) {
                int y = E[j];
                bool atomicPtr=visited2[y].load(std::memory_order_relaxed);
                if(!visited[y] && __sync_bool_compare_and_swap(
                                &visited[y],
                                false,
                                true)) {
                                    dist[y]=cur_dis+1;
                                    psflags[j-start] = 1;
                                }
                // else if(!atomicPtr && std::atomic_compare_exchange_strong(&visited2[y], &atomicPtr, true)) {
                //     psflags[j-start] = 1;
                //     dist[y]=cur_dis+1;
                //     //cout<<"Node Distance: seq for 3: "<<dist[y]<<endl;
                // }
            }
        }
        else {
            cilk_for(int j = start; j < end; j++) {
                int y = E[j];
                //bool atomicPtr=visited2[y].load(std::memory_order_relaxed);
                if(!visited[y] && __sync_bool_compare_and_swap(
                                &visited[y],
                                false,
                                true)) { 
                                    dist[y]=cur_dis+1;
                                    psflags[j-start] = 1;
                                }
            }
        }
        
        
        scan_exclusive(psflags, range, ceil(sqrt(range)));
        
        nghs[k] = filter(nghsVisitedLength, offset, E, psflags, start, end, x);

        frontierOffsets[k] = nghsVisitedLength;
    }
    int* effective_nghs=flatten(nghs, frontierOffsets, frontierLength);
    return effective_nghs;
}

int* filter(int &no_nghs_visited, int* offset, int* E, int* psflags, int start, int end, int node) {
    int range=end-start;
    no_nghs_visited = psflags[range-1];

    int* filtered_nghs = new int[no_nghs_visited];

    if (psflags[0]==1) filtered_nghs[0] = E[offset[node]];
    
    if (range < 8751) { 
        for (int i = start; i < end; i++) {
            if (psflags[i-start] != psflags[i-1-start])
                filtered_nghs[psflags[i-start] - 1] = E[i];
        }
    }
    else {
        cilk_for (int i = start; i < end; i++) {
            if (psflags[i-start] != psflags[i-1-start])
                filtered_nghs[psflags[i-start] - 1] = E[i];
        }
    }

    return filtered_nghs;
}


void scan_exclusive(int* ps, int n, size_t no_of_chunks) {
    if (n < 8100) {
        for (int i = 1; i < n; i++) ps[i] += ps[i-1];
        return;
    }
    size_t chunk_size = n / no_of_chunks;
    if (n % no_of_chunks != 0) chunk_size++;

    int* leftsum  = new int[no_of_chunks];
    leftsum[0] = 0;

    cilk_for (size_t i = 1; i < no_of_chunks; i++) {
        leftsum[i] = 0;
        for (size_t j = (i-1)*chunk_size; j < i*chunk_size; j++) { 
            if (j >= n) {
                break;
            } 
            leftsum[i] += ps[j];
        }
    }

    for (size_t i = 1; i < no_of_chunks; i++) {
        leftsum[i] += leftsum[i-1];
    }

    cilk_for (size_t i = 0; i < no_of_chunks; i++) {
        for (size_t j = 0; j < chunk_size; j++) {
            size_t oldIndex = j+chunk_size*i;
            if (oldIndex >= n) {
                break;
            }
            if (j == 0) {
                ps[oldIndex] += leftsum[i];
            }
            else {
                ps[oldIndex] += ps[oldIndex-1];
            }
        }
    }
}

int* flatten(int** nghs, int* frontierOffsets, int &oldFrontierLength) {
    int* newFrontierOffsets = new int[oldFrontierLength];
    cilk_for(size_t i = 0; i < oldFrontierLength; i++) {
        newFrontierOffsets[i] = frontierOffsets[i];
    }
    
    scan_exclusive(newFrontierOffsets, oldFrontierLength, ceil(sqrt(oldFrontierLength)));

    int newFrontierLength = newFrontierOffsets[oldFrontierLength-1];
    int* effective_nghs = new int[newFrontierLength];

    
    cilk_for(int i = 0; i < oldFrontierLength; i++) {
        int index;
        if (i == 0) index = 0;
        else index = newFrontierOffsets[i-1];
        cilk_for(int j = 0; j < frontierOffsets[i]; j++) {
            effective_nghs[index+j] = nghs[i][j];
        }
        
    }
    oldFrontierLength = newFrontierLength;

    return effective_nghs;
}

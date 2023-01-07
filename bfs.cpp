#include <iostream>
#include <cstdio>
#include <stdlib.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include "get_time.h"
#include "parse_command_line.h"
#include "bfs.h"
using namespace std;

#define NUM_ROUNDS 5

void seq_BFS(int n, int m, int* offset, int* E, int s, int* dist) {
	for (int i = 0; i < n; i++) dist[i] = -1;
	
	dist[s] = 0;
	int* q = new int[n];
	q[0] = s;
	int start = 0, end = 1;
	while (start != end) {
		int cur_node = q[start];
		int cur_dis = dist[cur_node];
		
		for (int i = offset[cur_node]; i < offset[cur_node+1]; i++) {
			int v = E[i];
			if (dist[v] == -1) {
				dist[v] = cur_dis + 1;
				q[end++] = v;
			}
		}
		start++;
	}
	cout << "traversed: " << end << " vertices" << endl;
}
	

int main(int argc, char** argv) {
	commandLine P(argc, argv,
		"[-g graph] [-s source]");
	string graphname = P.getOptionValue("-g", "soc-LiveJournal_sym.adj");
	int s = P.getOptionIntValue("-s", 214);
	
	string dir = "/usr/local/cs214/";
	string filename = dir+graphname;
	freopen(filename.c_str(), "r", stdin);
	
	int n, m;
	string graphtype; 
	cin >> graphtype;
	cin >> n;
	cin >> m;
	
	int* offset = new int[n+1];
	int* E = new int[m];
		
	cout << "reading graph" << endl;
	for (int i  = 0; i < n; i++) cin >> offset[i];
	for (int i  = 0; i < m; i++) cin >> E[i];
	offset[n] = m;
	cout << "finish reading graph" << endl;
	
	//correctness check, when you test performance, you can comment this out
	{
	int* dist_check = new int[n];
	timer tx; tx.start();
	
	seq_BFS(n, m, offset, E, s, dist_check);

	double seq_t = tx.stop();
	cout << "sequential time: " << seq_t << endl;
	
	int* dist = new int[n];
	
	BFS(n, m, offset, E, s, dist);
	
	for (int i = 0; i < n; i++) {
		if (dist[i] != dist_check[i]) {
			cout << "dist[i] " << dist[i] << "dist_check[i]: " << dist_check[i] << endl;
			cout << "wrong answer" << endl;
			exit(1);
		}
	}
	
	delete[] dist;
	delete[] dist_check;
	}
	
	{
	int* dist = new int[n];
	BFS(n, m, offset, E, s, dist);
	delete[] dist;
	}
	
	double tot_time = 0.0;
	for (int round = 0; round < NUM_ROUNDS; round++) {
		int* dist = new int[n];
		
		timer t; t.start();
		BFS(n, m, offset, E, s, dist);
		t.stop();
		double tm = t.get_total();
		
		cout << "BFS, Round " << round << ", time: " << tm << endl;
		tot_time += tm;

		delete[] dist;
	}
	cout << "BFS, Average time: " << tot_time/NUM_ROUNDS << endl;
	
	return 0;
}

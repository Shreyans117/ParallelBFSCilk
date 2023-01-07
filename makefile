CC = g++
CFLAGS = -O3 -mcx16 -march=native -DCILK -fcilkplus -std=c++17

all:	bfs

bfs:	bfs.cpp
	$(CC) $(CFLAGS) -DNDEBUG bfs.cpp -o bfs
	
	
clean:
	rm -f bfs

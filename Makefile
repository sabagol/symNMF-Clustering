CC=gcc
CFLAGS=-ansi -Wall -Wextra -Werror -pedantic-errors -std=c99

all: symnmf

symnmf: symnmf.c symnmf.h
	$(CC) $(CFLAGS) -o symnmf symnmf.c -lm

clean:
	rm -f *.o symnmf

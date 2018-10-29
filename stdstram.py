"""
Handling stdin stdout
Input stream have to be in formate:
N - demension of matrix
then matrix A:
| a11 ... a1n |
| a21 ... a2n |
|     .       |
|       .     |
| an1 ... ann |
thirdly b vector:
| b1 ... bn |
Finally C free coefficient:
C

See example test_input
"""

import sys


def getinputquadratic():
    input = sys.stdin.readlines()
    dimension = int(input[0])
    A = [[float(y) for y in x.split(' ')] for x in input[1:dimension+1]]
    b = [float(x) for x in input[dimension+1].split(' ')]
    C = input[-1]
    return [A, b, C]


if __name__ == "__main__":
    getinputquadratic()

import os 

# this if for the hungarian algorithm 
with open('arnie.txt', 'w+') as f: 
    f.write("linearpartition: . \nTMP: /tmp")

os.environ['ARNIEFILE'] = 'arnie.txt'

def mask_diagonal(matrix, mask_value=0):
    matrix=matrix.copy()
    n = len(matrix)
    for i in range(n):
        for j in range(n):
            if abs(i - j) < 4: 
                matrix[i][j] = mask_value
    return matrix
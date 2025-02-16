import numpy as np
import matplotlib.pyplot as plt

def linear_transform(A, v0, n):
    """Computes a sequence of vectors using the recursive formula v_k = A v_(k-1)."""
    vectors = np.zeros((2, n))
    vectors[:, 0] = v0

    for k in range(1, n):
        vectors[:, k] = np.matmul(A, vectors[:, k-1])

    return vectors

def plot_vectors(vectors):
    """Plots the generated vectors as points."""
    plt.scatter(vectors[0, :], vectors[1, :], s=1, color='green')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Transformations')
    plt.show()

# Example for part (a)
A = np.array([[0.5, -0.2], [0.3, 0.7]])
v0 = np.array([1, 1])
n = 100

vectors = linear_transform(A, v0, n)
plot_vectors(vectors)

# Part (b): Barnsley's Fern
def barnsley_fern(n):
    """Generates Barnsley's Fern using IFS transformations."""
    A1 = np.array([[0, 0], [0, 0.16]])
    A2 = np.array([[0.85, 0.04], [-0.04, 0.85]])
    A3 = np.array([[0.2, -0.26], [0.23, 0.22]])
    A4 = np.array([[-0.15, 0.28], [0.26, 0.24]])

    b1 = np.array([0, 0])
    b2 = np.array([0, 1.6])
    b3 = np.array([0, 1.6])
    b4 = np.array([0, 0.44])

    transformations = [(A1, b1, 0.01), (A2, b2, 0.85), (A3, b3, 0.07), (A4, b4, 0.07)]
    
    v = np.array([0, 0])  # Initial vector
    points = np.zeros((2, n))

    for i in range(n):
        r = np.random.rand()
        cumulative_prob = 0

        for A, b, p in transformations:
            cumulative_prob += p
            if r <= cumulative_prob:
                v = np.matmul(A, v) + b
                break
        
        points[:, i] = v

    return points

# Generate and plot Barnsley's Fern
fern_points = barnsley_fern(100000)
plt.scatter(fern_points[0, :], fern_points[1, :], s=0.1, color='green')
plt.axis("off")
plt.title("Barnsley's Fern")
plt.show()

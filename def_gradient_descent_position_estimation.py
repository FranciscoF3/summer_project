def gradient_descent_position_estimation(sensor_positions, measured_distances, initial_guess,
                                         max_iterations=1000, tolerance=1e-6):
    """
    Estimate the object's position using gradient descent.
    
    Parameters:
    - sensor_positions: numpy array of shape (m, 3)
    - measured_distances: numpy array of shape (m,)
    - initial_guess: numpy array of shape (3,)
    - learning_rate: float, step size for gradient descent
    - max_iterations: int, maximum number of iterations
    - tolerance: float, convergence threshold
    
    Returns:
    - estimated_position: numpy array of shape (3,), estimated object position
    """
    x = initial_guess.copy()
    m = sensor_positions.shape[0]
    cost_history = []
    residuals = []
    estm_pos = []
    for iteration in range(max_iterations):
        # Initialize gradient
        gradient = np.zeros_like(x)
        cost = 0.0
        a = []
        A = np.zeros((m, 3))
        d = np.zeros(m)
        for i in range(m):
            s_i = sensor_positions[i]
            r_i = measured_distances[i]
            
            # Vector from sensor to current estimate
            diff = x - s_i
            distance = np.linalg.norm(diff)
            #print(distance)
            
            # Avoid division by zero
            if distance == 0:
                print("bad zero")
                continue  # Or handle appropriately
             
            
            # Residual
            f_i = r_i - distance
            #print("\ndi:\n", f_i)
            
            d[i] = f_i
            
            # Unit vector
            u_i = diff / distance
            #print(u_i)
            
            A[i] = u_i

            #print(A)
            
            
           
        residuals.append(d)
        #print(A)
        # Solve the normal equations
        #print(np.linalg.matrix_rank(A))
        
        ATA = A.T @ A
        #print(ATA)
        #A_plus = np.linalg.pinv(A)
        A_plus_alt = np.linalg.inv(ATA) @ A.T

        # Solve for estimatded x
        #estimated_position = A_plus @ d
        estimated_position = A_plus_alt @ d

        #print(estimated_position)
        # Update position estimate
        x_new = x + estimated_position   
        estm_pos.append(x_new)
        # Check for convergence
        if np.linalg.norm(x_new - x) < tolerance:
            #print(f"Converged in {iteration + 1} iterations.")
            return x_new, cost_history, residuals, estm_pos, A
        
        # Update estimate
        x = x_new
        
    
   
    return x, cost_history, residuals, estm_pos, A

import numpy as np
import pandas as pd


S_Real = np.array([
    [1.778 + 0.035, -2.37 + 0.04, 1.994],
    [-.127 + 0.035,2.642 + 0.04, 2.121],
    [-2.07 + 0.035, 0.0508 + 0.04, 2.16]
])

df = pd.read_csv("C:/Users/Francisco/Documents/03_Work/01_Projects/02_PNTi/variance_test_04_22_25/p4.csv")
measured_distances = df[['Tag1_m_', 'Tag2_m_', 'Tag3_m_']].values  # Meters
# z_guess = 0.1  # fixed height like UWB tag on robot
initial_guess = np.array([1.0, 1.0, 0.1])

estimated_positions = []

for distances in measured_distances:
    est_pos, _, _, _, _ = gradient_descent_position_estimation(
        sensor_positions=S_Real,
        measured_distances=distances,
        initial_guess=initial_guess
    )
    estimated_positions.append(est_pos)

estimated_positions = np.array(estimated_positions)


# Create a DataFrame with columns X, Y, Z
estimated_df = pd.DataFrame(estimated_positions, columns=["X (m)", "Y (m)", "Z (m)"])

# Save to CSV
estimated_df.to_csv("C:/Users/Francisco/Documents/03_Work/01_Projects/02_PNTi/variance_test_04_22_25/p4_position.csv", index=False)


import matplotlib.pyplot as plt

"""""
plt.plot(estimated_positions[:, 0], estimated_positions[:, 1], label='Estimated Path')
plt.xlabel('X position (m)')
plt.ylabel('Y position (m)')
plt.title('Estimated 2D Path from UWB Trilateration')
plt.legend()
plt.grid()
plt.axis('equal')
plt.show()

"""


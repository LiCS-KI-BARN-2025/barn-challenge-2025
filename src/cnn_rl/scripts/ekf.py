from filterpy.kalman import ExtendedKalmanFilter
import numpy as np

class DiffDriveEKF(ExtendedKalmanFilter):
    def __init__(self, dt):
        # State: [x, y, theta, v, w]
        super().__init__(dim_x=5, dim_z=2)  # Changed dim_z to 2 for position only
        self.dt = dt
        
        # Initial uncertainty
        self.P *= 1000
        
        # Process noise - higher for theta since it's only predicted
        self.Q = np.diag([0.1, 0.1, 0.3, 0.2, 0.2])
        
        # Measurement noise - only for position
        self.R = np.diag([0.1, 0.1])
        
    def predict(self, u=None):
        # Use input controls if provided, otherwise use current state velocities
        if u is not None:
            v = u[0]  # linear velocity
            w = u[1]  # angular velocity
            # Update velocity states
            self.x[3] = v
            self.x[4] = w
        else:
            v = self.x[3,0]
            w = self.x[4,0]
        
        # State transition function
        x, y, theta = self.x[0:3].flatten()
        
        if abs(w) < 1e-6:
            # Straight motion
            self.x[0] += v * np.cos(theta) * self.dt
            self.x[1] += v * np.sin(theta) * self.dt
        else:
            # Circular motion
            self.x[0] += (v/w) * (np.sin(theta + w*self.dt) - np.sin(theta))
            self.x[1] += (v/w) * (-np.cos(theta + w*self.dt) + np.cos(theta))
            
        self.x[2] += w * self.dt
        
        # Calculate Jacobian F
        F = np.eye(5)
        if abs(w) < 1e-6:
            F[0,2] = -v * np.sin(theta) * self.dt
            F[1,2] = v * np.cos(theta) * self.dt
        else:
            F[0,2] = (v/w) * (np.cos(theta + w*self.dt) - np.cos(theta))
            F[1,2] = (v/w) * (np.sin(theta + w*self.dt) - np.sin(theta))
            
        self.F = F
        super().predict()
        
    def HJacobian(self, x):
        # Measurement model Jacobian - only for position
        H = np.zeros((2, 5))
        H[0,0] = 1  # x
        H[1,1] = 1  # y
        return H
        
    def Hx(self, x):
        # Measurement model - only return position
        return np.array([x[0], x[1]]).reshape(2,1)

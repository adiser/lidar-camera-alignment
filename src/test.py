import torch
import torch.nn as nn
import torch.optim as optim

# Define a model using a quaternion for rotation
class QuaternionRotationModel(nn.Module):
    def __init__(self):
        super(QuaternionRotationModel, self).__init__()
        # Initialize quaternion with small random values
        self.quaternion = nn.Parameter(torch.randn(4) * 0.01)

    def forward(self):

        # Quaternion to rotation matrix conversion
        # q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
        return self.quaternion


if __name__ == "__main__":
    # Instantiate the model
    model = QuaternionRotationModel()

    # Define a loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Target rotation matrix (identity matrix for example)
    target_rotation = torch.eye(3)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        rotation_matrix = model()
        loss = criterion(rotation_matrix, np.ones(4))
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
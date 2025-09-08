# Test HGT import
try:
    from torch_geometric.nn import HGTConv, Linear
    print("PyG imports successful")
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    print("PyTorch imports successful")
    
    # Define a simple test class
    class TestHGT(nn.Module):
        def __init__(self):
            super().__init__()
            print("TestHGT initialized")
    
    test_model = TestHGT()
    print("Test model created successfully")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

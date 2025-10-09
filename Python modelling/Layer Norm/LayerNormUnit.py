import numpy as np

class LayerNormUnit:
    def __init__(self, embedding_dim, epsilon=1e-5):
        self.embedding_dim = embedding_dim
        self.epsilon = epsilon
        # Learnable parameters (initialized like standard LN)
        self.gamma = np.ones((embedding_dim,), dtype=np.float32)
        self.beta = np.zeros((embedding_dim,), dtype=np.float32)
        # Control signals
        self.start = False
        self.done = False

    def forward(self, x, non_linearity_unit=None):
        """
        Simulated LayerNorm hardware unit.

        Args:
            x : np.ndarray
                Shape [sequence_length, embedding_dim]
            non_linearity_unit : function
                Optional function to apply before normalization (e.g., ReLU)
        Returns:
            y : np.ndarray or None
        """
        if not self.start:
            print("LayerNorm waiting for start signal ...")
            return None

        # Step 1: Apply non-linearity if provided
        if non_linearity_unit is not None:
            x = non_linearity_unit(x)

        # Step 2: Compute mean & variance across embedding dimension
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)

        # Step 3: Normalize
        x_hat = (x - mean) / np.sqrt(var + self.epsilon)

        # Step 4: Scale and shift
        y = self.gamma * x_hat + self.beta

        # Step 5: Raise done flag
        self.done = True
        return y


#################### DEMO TEST ####################
if __name__ == "__main__":
    seq_len, embed_dim = 4, 8
    lnu = LayerNormUnit(embed_dim)

    # Dummy activations
    activations = np.random.randn(seq_len, embed_dim).astype(np.float32)

    # Example non-linearity (ReLU)
    relu = lambda x: np.maximum(0, x)

    # Start the block
    lnu.start = True
    output = lnu.forward(activations, non_linearity_unit=relu)

    if lnu.done:
        print("✅ LayerNorm Done")
        print("Input:\n", activations)
        print("Output:\n", output)

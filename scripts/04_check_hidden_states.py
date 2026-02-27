import torch

PATH = r"G:\LLM\experiment\data\hidden_states_minimal.pt"

def main():
    x = torch.load(PATH)  # shape: [tasks, layers, hidden]
    print("Loaded:", PATH)
    print("Shape:", tuple(x.shape))
    print("Dtype:", x.dtype)

    # Simple sanity stats
    print("Mean:", x.mean().item())
    print("Std:", x.std().item())

    # Check differences between layers: average L2 norm per layer (across tasks)
    # x: [T, L, H] -> norms: [L]
    norms = torch.linalg.vector_norm(x, dim=2).mean(dim=0)
    print("Layer norm stats:")
    print("  min:", norms.min().item(), "max:", norms.max().item())
    print("  first 5 layers:", norms[:5].tolist())
    print("  last 5 layers:", norms[-5:].tolist())

if __name__ == "__main__":
    main()
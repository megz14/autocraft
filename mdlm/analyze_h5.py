# analyze_h5.py
import argparse
import h5py
import numpy as np

def main(path, dataset="voxels", n=4):
    with h5py.File(path, "r") as f:
        data = f[dataset]
        print(f"{dataset} shape:", data.shape)
        print("dtype:", data.dtype)

        # basic stats
        arr = data[...]
        print("min/max:", arr.min(), arr.max())
        unique, counts = np.unique(arr, return_counts=True)
        print("unique values (truncated):")
        for val, cnt in zip(unique[:20], counts[:20]):
            print(f"  {val}: {cnt}")

        # show a small slice
        nz = np.argwhere(arr != 0)
        if nz.size:
            i, j, k = nz[0]
            print(f"first non-zero at ({i}, {j}, {k}) -> {arr[i,j,k]}")
        print("sample block:\n", arr[:n, :n, :n])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="dataset/test.h5")
    parser.add_argument("--dataset", default="voxels")
    parser.add_argument("--n", type=int, default=4)
    args = parser.parse_args()
    main(args.path, args.dataset, args.n)
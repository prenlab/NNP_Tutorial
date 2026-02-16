# NNP Tutorial: Building an AMOEBA+NN Hybrid Potential 

A complete workflow tutorial for building a neural network potential (NNP), covering data generation, model training, and model evaluation. This tutorial focuses on creating a **correction term** for the AMOEBA force field (specifically for Copper ions), but the principles apply to general hybrid force field development.


## Tutorial Walkthrough 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prenlab/NNP_Tutorial/blob/main/nnp_training.ipynb)

This repository contains the `nnp_training.ipynb` notebook, which guides you through the lifecycle of an NNP project. Open it in Colab 
 to train and test the neural network. Below is a detailed text walkthrough of the important concepts covered in this tutorial.

---

### 1. Data Preparation
The foundation of the NNP is the dataset. We use a structured format linking metadata (CSV) with heavy tensor data (HDF5).

#### The Dataset Structure
*   **Molecule ID (`ID`):** A unique string identifier for a specific molecular cluster (e.g., `md6n` or `mc6n` in this tutorial). This ID links the rows in the CSV file to the groups in the HDF5 file.
*   **Conformer ID (`CONF_ID`):** An integer representing a specific snapshot/conformation of that molecule.

#### File 1: Metadata (`data.csv`)
This file controls the training logic.
| Column | Meaning |
| :--- | :--- |
| `ID` | The molecule name. Must match the group key in the HDF5 file. |
| `CONF_ID` | The specific geometry index. |
| `dft_energy` | The reference Quantum Mechanical energy (e.g., MP2, wB97X). |
| `amoeba_energy` | The baseline classical energy calculated by Tinker. |
| **`label_e`** | **The Training Target.** Defined as $U_{target} = U_{QM} - U_{AMOEBA}$ in this tutorial. The NN learns this residual error. |
| `division` | An integer (0-9) used for **Train/Val splitting**. We split by molecule or arbitrary division index to ensure the model generalizes rather than memorizing specific conformers. |

#### File 2: Tensor Data (`data.h5`)
A hierarchical database (HDF5) storing the physical properties.
*   `/Group_Name/coordinates`: Shape `(N_confs, N_atoms, 3)`. The XYZ positions in Angstroms.
*   `/Group_Name/atomic_numbers`: Shape `(N_confs, N_atoms)`. (e.g., 29 for Cu, 7 for N).
*   `/Group_Name/forces`: Shape `(N_confs, N_atoms, 3)`. The QM-MM forces (required if training with force regularization).

---

### 2. Neural Network Basics
The notebook uses the `amoeba_nn` package, which implements an ANI-style architecture. Here are the core concepts configured in the `config.yaml` and code:

#### Atomic Environment Vectors (AEV)
It is not good to let the NN "see" Cartesian coordinates directly because they change with rotation/translation. Instead, we convert coordinates into **AEVs**.
*   **Radial Part:** Encodes distances between atom pairs.
*   **Angular Part:** Encodes angles between atom triplets.
*   **Hyperparameters:** Settings like `radial_cutoff` (e.g., 3.5 Å for metals, 5.2 Å for organics) determine how far the NN "sees".

#### Network Architecture
We use element-specific Multi-Layer Perceptrons (MLPs).
*   **Structure:** Input (AEV) $\rightarrow$ Hidden Layers (e.g., 128, 64, 32 nodes) $\rightarrow$ Output (Atomic Energy).
*   **Activation:** CELU functions are used for non-linearity.
In this tutorial, we will train one MLP for Copper ions.

#### Loss Function Design
We train on a composite loss function to ensure physical stability:
$$Loss = MSE(Energy) + \alpha \times MSE(Forces)$$
In addition to energies, training on forces ($-\nabla E$) ensures the potential energy surface is smooth, which is critical for stability in MD simulations.

---

### 3. Training Process
The notebook executes the `amoeba_nn/run.py` script. Key training hyperparameters to watch include:

*   **Learning Rate (`lr`):** How fast the model updates weights.
*   **LR Schedule (`lr_patience`, `lr_factor`):** If the validation loss stops improving (plateaus), we reduce the learning rate to fine-tune the model.
*   **Early Stopping:** If the model stops improving entirely, training halts to prevent overfitting.

**Monitoring:** The notebook utilizes **TensorBoard**. You should look for:
1.  **Convergence:** The training and validation loss curves should decrease and level off.
2.  **Overfitting:** If Training Loss keeps dropping but Validation Loss rises, the model is memorizing data instead of learning physics.

---

### 4. Evaluation

Once trained, the PyTorch model (`.pt`) can be converted to a `.prm` for Tinker9 to run simulation with.

1.  **Conversion:** Use the `pt2prm.py` script in the notebook to extract weights from the `.pt` file and write them into a Tinker parameter file (`.prm`).
2.  **Geometry Optimization:** The notebook runs `tinker9 minimize`.
    *   *Goal:* We verify if the hybrid model reproduces the **Jahn-Teller effect** (asymmetric bond lengths in Cu complexes) which classical models fail to capture. That is our design goal for this NN model.
3.  **Molecular Dynamics (Coming Soon):** Finally, we run a short MD simulation to ensure the system is stable and does not "explode" due to unphysical forces.

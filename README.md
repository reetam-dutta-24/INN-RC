# INN-RC: Lossless Image Steganography with a Message-Conditioned Integer Coupling Network

UCT702 research project, track **TRK-07 (Lossless End-to-End Steganography)**.

A secret message is hidden in a grayscale image by a small integer coupling network. On a lossless
channel the receiver gets back **both the message (BER = 0) and the cover image, bit for bit**. If the
stego image is distorted (noise, JPEG) the cover is lost, but most of the message can still be read.

## How it works (short version)

1. The message is scrambled with a logistic-map keystream and spread over `s x s` blocks.
2. The cover is split into two pixel halves. Four coupling steps add `round(phi(other half, message))`
   to one half at a time. All steps are integer additions conditioned on the message, so once the
   receiver knows the message it can subtract them again exactly. No latent variable is produced.
3. A small CNN decoder reads the message back from the stego.
4. Residual compensation: the bits the decoder gets wrong and the pixels clipped at 0/255 are coded
   with an adaptive arithmetic coder (contexts = decoder confidence and neighbourhood) and hidden in
   the stego with multi-layer rhombus PEE.
5. The receiver undoes PEE, reads the message, fixes the wrong bits with the side information and runs
   the couplings backwards to get the cover.

## Folder layout

```
data/        train_urban100 (training), test_bsd100, test_kodak, test_medical (DICOM -> 8 bit PNG)
training/    PyTorch models + training scripts (CPU is enough)
models/      trained weights (.mat, read directly by MATLAB) and training logs
matlab/      everything else: inference, coding, PEE, embed/extract, benchmarks, GUI
results/     CSV files written by the benchmark scripts, example images
tools/       make_figures.py (tables/plots for the paper from the CSV files)
paper/       IEEE conference paper (LaTeX)
docs/        Phase 1 literature review
```

## Running it in MATLAB

Tested functions only need base MATLAB plus the Image Processing Toolbox (for `imwrite`/JPEG).
The GUI uses App Designer components (`uifigure`), so R2020b or newer is recommended.

```matlab
cd matlab
INNRC_App                                % GUI: load image, embed, extract, try a lossy channel

% one image from the command line
net = icinn_load('../models/icinn.mat');
I   = imread('../data/test_kodak/kodim23.png');
s   = 4;                                 % 1024 bits on a 256x256 image
msg = rand(numel(I)/4/s^2, 1) > 0.5;
[S, info] = inncr_embed(I, msg, net, s, 12345);
[m2, I2]  = inncr_extract(S, net, s, 12345);
isequal(I2, I), isequal(m2, msg)         % both true

run_benchmark                            % all 132 test images, writes results/benchmark.csv
run_attacks                              % robustness test, writes results/attacks.csv
make_visuals                             % example images for the paper
```

## Training

```bash
cd training
pip install torch numpy scipy pillow
./run_all_training.sh                    # both IC-INN models + the float INN baseline
```

`matlab/train_icinn_matlab.m` is a Deep Learning Toolbox version of the same training loop
(`dlarray`, `dlgradient`, `adamupdate`); it saves weights in the same format.

## Datasets

| Set | Images | Use | Source |
|---|---|---|---|
| Urban100 | 100 (92 train / 8 val) | training | Huang et al., CVPR 2015 |
| BSD100 | 100, 256x256 centre crop | test | Martin et al., ICCV 2001 |
| Kodak | 24, grey, 512 crop -> 256 | test | r0k.us/graphics/kodak |
| DICOM | 8 CT/MR/CR slices -> 256 | test | github.com/pydicom/pydicom-data |

BOSSbase 1.01 was not reachable from our network. `run_benchmark` reads every PNG in a test folder,
so BOSSbase images can be dropped into `data/` and run the same way.

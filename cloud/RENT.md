# Renting a GPU to train the model — step by step

The whole run costs roughly **$3–15** and finishes in a **few hours** on a
mid-range GPU. You create the account and rent the box (I can't enter payment
details for you); everything else is one paste-command.

## Which GPU
You do **not** need an A100/H100. A single **RTX 4090 (24 GB)** or **A40** is
plenty for this ~20M-param model. Pick the cheapest available.

## Where to rent — pick one

### RunPod (easiest — recommended for a one-off)
1. Sign up at **runpod.io**, add ~$15 credit.
2. **Deploy → Pods → GPU**, choose an **RTX 4090** (Community Cloud = cheapest,
   ~$0.34–0.70/hr).
3. Template: **"RunPod PyTorch 2.x"** (has CUDA + torch preinstalled).
   Set container disk / volume to **~40 GB** (datasets need room).
4. Deploy, then open the pod's **Web Terminal** (or Jupyter → Terminal).

### Vast.ai (cheapest, slightly more fiddly)
1. Sign up at **vast.ai**, add credit.
2. Search an **RTX 4090**, filter for a **PyTorch** image, ≥40 GB disk.
3. Rent, then open its terminal / SSH.

### Lambda Labs (simplest billing, a bit pricier)
- **lambdalabs.com** → launch an **A10 / A100** instance with the PyTorch image,
  SSH in. ~$0.75–1.10/hr.

## Then — the one command
In the pod's terminal, paste:

```bash
git clone https://github.com/SmrtKraft/metagraphe-train && cd metagraphe-train && bash cloud/bootstrap.sh
```

That re-downloads the public datasets on the pod, packs them, and trains on the
GPU (prints F1 each epoch). It runs unattended — leave the tab open (or use
`tmux`/`nohup` so it survives a disconnect: `tmux new -s train` then paste, and
`Ctrl-b d` to detach).

## When it's done
The trained model is `checkpoints/ft_epochNN.pth` (~170 MB). Download the
newest one:
- **RunPod:** the file browser, or `runpodctl send checkpoints/ft_epoch50.pth`.
- **Vast/Lambda:** `scp` it, or upload to a transfer service and paste me the link.

**Then STOP / terminate the pod** so billing ends. Send me the checkpoint and I
convert it to ONNX and wire it into KANON as the new engine.

## Cost sanity check
RTX 4090 at ~$0.5/hr × ~4–8 h ≈ **$2–4**. Even a slow, generous run stays under
~$15. Terminate when done — idle pods keep charging.

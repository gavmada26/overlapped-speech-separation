# Intelligent Overlapped Speech Separation & Audio Enhancement Platform

<p align="center">
  <img src="assets/logo.png" alt="Project Logo" width="160" height="160">
</p>

<p align="center">
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.9%20%7C%203.10-blue?style=for-the-badge&logo=python" alt="Python"></a>
  <a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://gradio.app"><img src="https://img.shields.io/badge/Gradio-Frontend-orange?style=for-the-badge" alt="Gradio"></a>
  <a href="https://speechbrain.github.io"><img src="https://img.shields.io/badge/SpeechBrain-Framework-purple?style=for-the-badge" alt="SpeechBrain"></a>
</p>

## 📌 Introduction & Core Objective
This repository contains a high-performance audio signal processing application designed to solve the classic acoustic **"Cocktail Party Problem"**—the digital isolation of individual speaker signals from single-channel overlapped audio mixtures (Blind Source Separation - BSS).

The platform bridges the gap between classical Digital Signal Processing (DSP) techniques and modern Deep Learning frameworks. It ingests complex acoustic streams, performs real-time vocal boundary scanning, maps multi-speaker overlaps, eliminates stationary environmental noise, and calculates objective evaluation metrics along with non-intrusive neural quality predictions.

---

## 🏗️ System Architecture & Workflow Diagram

The platform routes multi-channel and single-channel audio mixtures through a modular execution pipeline optimized for processing throughput, mathematical precision, and low-latency rendering:

```mermaid
graph TD
    %% Audio Input Layer
    Input[Mixed Audio Input Stream / .wav] --> VAD[Silero Voice Activity Detection]
    
    %% Frame Trimming and Standardization
    VAD -->|Active Speech Timestamps| Stitch[Dynamic Segment Stitching]
    VAD -->|Silent Intervals Dropped| Resample[TorchAudio Resampling Core]
    Stitch --> Resample
    
    %% Separation Routing Core
    Resample -->|Standardized Uniform 16 kHz Signal| CoreRouting{Neural Core Selector}
    
    %% Deep Learning Execution Chains
    CoreRouting -->|Scenario 1: Multi-Speaker Isolation| SepFormer[SpeechBrain SepFormer Models]
    CoreRouting -->|Scenario 2: Audio Dissection & Stems| Demucs[Meta Demucs Hybrid WaveNet Engine]
    
    %% DSP Subtraction Layer
    SepFormer --> DSP[DSP Post-Processing: Non-Linear NoiseReduce]
    Demucs --> DSP
    
    %% Target Interfaces
    DSP --> Analytics[Advanced Quality Diagnostics Suite]
    DSP --> Interface[Gradio Graphical Dark-Mode Frontend]
    
    %% Evaluation Parameters
    Analytics -->|Objective Mathematical Scores| Metrics[SI-SDR / PESQ / ESTOI / SIM / DNSMOS ITU-T P.808]

    %% Formatting Nodes Styles
    style Input fill:#2c3e50,stroke:#34495e,stroke-width:2px,color:#fff
    style VAD fill:#7f8c8d,stroke:#95a5a6,stroke-width:2px,color:#fff
    style CoreRouting fill:#d35400,stroke:#ba4a00,stroke-width:2px,color:#fff
    style SepFormer fill:#2980b9,stroke:#2471a3,stroke-width:2px,color:#fff
    style Demucs fill:#16a085,stroke:#117a65,stroke-width:2px,color:#fff
    style DSP fill:#27ae60,stroke:#1e8449,stroke-width:2px,color:#fff
    style Analytics fill:#8e44ad,stroke:#7d3c98,stroke-width:2px,color:#fff
    style Interface fill:#2c3e50,stroke:#34495e,stroke-width:2px,color:#fff

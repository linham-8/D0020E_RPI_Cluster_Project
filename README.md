# D0020E Raspberry PI Cluster Project

## Introduction
During the last years, AI-models such as ChatGPT, Gemini and Grok have gone from being something new and revolutionary, to something commonly used and well known.

Modern deep-learning often depends on some type of parallelism technique to be able to train complex models on large amounts of data. Examples of these are data parallelism, model parallelism, expert parallelism and pipeline parallelism, all of which have been implemented in this project. These techniques are often used on several powerful GPUs that are physicalle close to each other, to minimise latency and increase throughpu. There are tools available to run training using parallelism, such as the PyTorch library.

Being able to compare the different types of parallelism can provide a better understanding of where performance might be improved and what type of parallelism works best for the setup being tested. The reason why Raspberry Pis are a good choice for the project is their low cost, high availability and energy-efficiency. This makes it possible to build and run a distributed AI cluster without having to use expensive and less commonly available hardware such as discrete GPUs. The limited system resources also make it easier to see how training and running of different models affect performance, without having to use larger datasets. It also makes it easier to analyse and understand the behaviour of the system, depending on which model runs.

What is missing is a tool to benchmark these distributed systems. A tool that shows the performance of both hardware and software in real-time, and then allows saving and summarising the results. This project provides that, with a modular testbed that allows direct comparison of different parallelism techniques and their effect on the system, all through an easy to use web-interface.

The project is built with python and PyTorch, using the GLOO-backend to synchronise processes across the nodes.

## Goals
The primary goals of the project are:

- Build and implement a Raspberry Pi Cluster capable of running distributed PyTorch models
- Implement and test multiple types of parallelism, such as expert, model, and data parallelism
- Develop an interface to easily switch between these and benchmark these parallelism techniques
- Collect and analyse performance data, such as throughput, latency, and efficiency
- Document the implementation process, challenges, and insights from the experiments

## Cluster setup
Instructions on how to setup the cluster can be found [here](instructions.md).

## Requirements and dependencies
All of the used dependecies and versions can be found [here](requirements.md).
To download the dependencies, run:
```pip install -r requirements.txt```

# SMWLevelGenerator

## Introduction

SMWLevelGenerator is a level generation framework for Super Mario
World based on deep neural networks.  
By combining techniques from natural language processing, generative
methods and image processing, we obtain a level generation pipeline.

We provide setup scripts, database generation, multiple models and
their corresponding training loops. When you actually want to generate
levels, we have functions handling everything in the background.

For fundamental questions, check out the
[thesis](thesis/SMWLevelGenerator.pdf).

## Table of Contents
   - [Features](#features)
   - [Setup](#setup)
      - [Dependencies](#dependencies)
      - [Quick Start](#quick-start)
      - [Environment](#environment)
      - [Databases](#databases)
         - [Generate the Databases Yourself](#generate-the-databases-yourself)
         - [Download Pre-computed Databases](#download-pre-computed-databases)
   - [Training](#training)
      - [Sequence Prediction](#sequence-prediction)
      - [First Screen Generation](#first-screen-generation)
      - [Metadata Prediction](#metadata-prediction)
   - [Generation](#generation)
      - [Whole Levels](#whole-levels)
      - [Predictions Only](#predictions-only)
      - [First Screens](#first-screens)
   - [Loading Models](#loading-models)
      - [JLD2](#jld2)
      - [BSON](#bson)
   - [License](#license)

## Features

Three different tasks:
   - Sequence prediction: Predict the rest of a level from one or more
     inputs.
   - First screen generation: Generate the first screen of a level
     from random noise.
   - Metadata prediction: Predict a level's metadata from its first
     screen.

For each task, we implement multiple models and a training loop. The
training loop works on any given model if it implements our
`LearningModel` interface.

Models work on data in different dimensionalities:
   - 1D: a line of the level
   - 2D: one layer of the level
   - 3D: multiple layers of the level

3-dimensional data can have different sizes in the third dimension
(meaning any combination of layers).

All models handle all dimensionalities correctly if setup correctly.
For some dimensions, we provide the correct setups. Any others may
easily be specified manually; we focus on easy extension.

## Setup

### Dependencies

- [Bash](https://www.gnu.org/software/bash/) (not required if you
  download pre-computed databases; see below)
- Any decompression program (we recommend
  [7-Zip](https://www.7-zip.org/))
- [Julia](https://julialang.org/) (version 1.2 or 1.3+)
- [TensorBoard](https://www.tensorflow.org/) (will be made optional)
- [Wine](https://www.winehq.org/) (if not on Windows; required to run
  [Lunar Magic](https://fusoya.eludevisibility.org/lm/index.html))

Execute the following in your command line of choice in this
directory:

```sh
julia --project -e 'using Pkg; Pkg.instantiate()'
```

Or in the Julia REPL, execute:

```julia
julia> ] activate .; instantiate
```

Julia is now correctly setup for this project with all dependencies
and their correct versions. Whenever you execute Julia for this
project (in this repository), start it as `julia --project`
(optionally, to set optimization to high, append ` -O3` to the
previous command).

With the dependencies setup, you can just skip all the boring manual
setup and get right to [downloading pre-computed
databases](#download-pre-computed-databases).

### Quick Start

Download and unzip
[this](https://drive.google.com/uc?export=download&id=1pj7HZOHiZwllOlxBYQaHksmuVHfQc5Ab),
then see [Training](#training) or [Generation](#generation).

### Environment

Currently, we only provide shell scripts (Bash compatibility) for
the basic setup. If you do not have access to a shell script
interpreter, either wait for the Julia implementation or use the
pre-computed databases.  
The most convenient setup command is the following:

```bash
./scripts/setup.sh
```

To download from the original source (not recommended to save SMW
Central's servers from stress; also may not work due to DDoS
protection and/or download limits):

```bash
./scripts/setup_smwcentral.sh
```

Alternatively, check out the [scripts](scripts) folder to find out how
to set everything up manually. We have documentation!

### Databases

You may either generate the databases yourself or download them.

#### Generate the Databases Yourself

After the previous setup, execute the following:

```
julia --project -e 'using SMWLevelGenerator; generate_default_databases()'
```

... or execute it in the Julia REPL if Julia is properly activated in
the project.

Or checkout the documentation to [`generatedb`](src/data/database.jl)
and related functions to generate databases individually.

If there were issues with one of the previous steps, you can still
generate the databases yourself by downloading the raw dumped level
files as a 8 MB `7z` from
[here](https://drive.google.com/uc?export=download&id=1Nq4eiBNEkJm9Eq2FWDrMAAftFNf4GjX3)
or as a 23 MB `tar.gz`
[here](https://drive.google.com/uc?export=download&id=1FILRqUM2MEjLXo9UZ_xFnuIREFg_liUA).
You still need to decompress these in the main folder, then the above
line will work.

#### Download Pre-computed Databases

Alternatively, download a 34 MB `7z` file from
[here](https://drive.google.com/uc?export=download&id=1pj7HZOHiZwllOlxBYQaHksmuVHfQc5Ab)
or a 60 MB `tar.gz` from
[here](https://drive.google.com/uc?export=download&id=19KMFjeZFIzFPannYSMkLZAHE32YL1Ezc).
You will need to decompress these.

## Training

For training, you can give parameters to the model as well as to the
training loop. Your first decision should be which dimensionality of
data to train on. Other parameters can be found in one of the three
individual training loops ([sequence
prediction](src/learning/training_loop.jl), [generative
methods](src/learning/gan_training.jl) and [image
processing](src/learning/meta_training.jl)).

Now, let's look at some example configurations for 2-dimensional data
(these do not include all models). To find all supplied convenience
constructors for the models, take a look at the `export`ed functions
in [SMWLevelGenerator.jl](src/SMWLevelGenerator.jl).

### Sequence Prediction

For an LSTM:

```sh
julia --project -O3 -e '
    using SMWLevelGenerator;
    const model = lstm2d();
    const dbpath = "levels_2d_flags_t.jdb";
    const res = trainingloop!(model, dbpath, TPs(
        epochs=1000,
        logdir="exps/meta_2d_low_learning_rate",
        dataiter_threads=3,
        use_soft_criterion=false,
    ));
'
```

For a GPT-2-based transformer using the soft criterion:

```sh
julia --project -O3 -e '
    using SMWLevelGenerator;
    const model = transformer2d();
    const dbpath = "levels_2d_flags_t.jdb";
    const res = trainingloop!(model, dbpath, TPs(
        epochs=1000,
        logdir="exps/meta_2d_low_learning_rate",
        dataiter_threads=3,
        use_soft_criterion=false,
    ));
'
```

### First Screen Generation

For a standard DCGAN:

```sh
julia --project -O3 -e '
    using SMWLevelGenerator;
    const d_model = discriminator2d(64);
    const g_model = generator2d(64, 96);
    const dbpath = "levels_2d_flags_t.jdb";
    const res = gan_trainingloop!(d_model, g_model, dbpath, GTPs(
        epochs=1000,
        lr=5e-5,
        logdir="exps/gan_2d",
        dataiter_threads=3,
    ));
'
```

For a fully connected Wasserstein GAN using RMSProp instead of Adam:

```sh
julia --project -O3 -e '
    using SMWLevelGenerator;
    import Flux;
    const d_model = densewsdiscriminator2d(64, 4);
    const g_model = densewsgenerator2d(64, 4, 96);
    const dbpath = "levels_2d_flags_t.jdb";
    const res = gan_trainingloop!(d_model, g_model, dbpath, GTPs(
        epochs=1000,
        lr=5e-5,
        optimizer=Flux.RMSProp,
        d_warmup_steps=0,
        d_steps_per_g_step=1,
        logdir="exps/densewsgan_2d_rmsprop",
        dataiter_threads=3,
    ));
'
```

### Metadata Prediction

For a convolutional metadata predictor:

```sh
julia --project -O3 -e '
    using SMWLevelGenerator;
    const model = metapredictor2d(64);
    const dbpath = "levels_2d_flags_t.jdb";
    const res = meta_trainingloop!(model, dbpath, MTPs(
        epochs=1000,
        lr=5e-5,
        logdir="exps/meta_2d_low_learning_rate",
        dataiter_threads=3,
    ));
'
```

For a fully connected metadata predictor:

```sh
julia --project -O3 -e '
    using SMWLevelGenerator;
    const model = densemetapredictor2d(64, 3);
    const dbpath = "levels_2d_flags_t.jdb";
    const res = meta_trainingloop!(model, dbpath, MTPs(
        epochs=1000,
        logdir="exps/densemeta_2d_64_3",
        dataiter_threads=3,
    ));
'
```

## Generation

We will only list some examples. Once again, please take a look at the
`export`ed functions in
[SMWLevelGenerator.jl](src/SMWLevelGenerator.jl) to get a better idea
of what is possible.

### Whole Levels

The following writes 16 levels to a ROM (not necessarily all
differently numbered, meaning the same level may be overwritten by a
later generation).

```julia
julia> rompath = joinpath("path", "to", "My Modifiable ROM.smc");

julia> writelevels(predictor, generator, metapredictor,
                   inputs=16, write_rom=rompath);
```

### Predictions Only

Predict a whole hack based on the first columns of each level in the
vanilla game:

```julia
julia> rompath = joinpath("path", "to", "My Modifiable ROM.smc");

julia> predict_vanilla(predictor, db, first_screen=false, write_rom=rompath);
```

### First Screens

Generate a single screen for levels 0x105 and 0x106:

```julia
julia> rompath = joinpath("path", "to", "My Modifiable ROM.smc");

julia> writescreen(generator, write_rom=rompath);

julia> writescreen(generator, write_rom=rompath, number=0x106);
```

## Loading Models

To load a model, you need to `import` or be `using` all the packages
the model uses as well (good candidates are `Flux` and/or
`Transformers`).

### JLD2

Due to a bug in JLD2, you will additionally need to `import` the
module containing any GANs you may want to load. For example, for a
stored `WassersteinGAN`, do the following:

```julia
julia> using SMWLevelGenerator, Flux, JLD2

julia> import SMWLevelGenerator.WassersteinGAN  # Change this line if necessary.

julia> generator = jldopen("exps/my_best_wsgan_checkpoint.jld2") do cp
           # This _must_ not give a warning like "Warning: type ... does
           # not exist in workspace; reconstructing"!
           generator = cp["g_model"]  # Note that the keys are `String`s.
       end;
```

Other models do not need this extra line.

### BSON

These checkpoints are easier to use but are less stable than the JLD2
checkpoints, meaning you may experience unrecoverable checkpoints (in
other words, data loss). For these, you do something like this:

```julia
julia> using SMWLevelGenerator, Flux, BSON

julia> cp = BSON.load("exps/my_best_wsgan_checkpoint.jld2");
Dict{Symbol,Any} [...]

julia> generator = cp[:g_model];  # Note that the keys are `Symbol`s.
```

## License

SMWLevelGenerator - Generating Super Mario World Levels Using Deep Neural Networks  
Copyright (C) 2019 Jan Ebert

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; only version 2
of the License (GPL-2.0-only).

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

Contact:
```
domain.de | local-part
----------+---------------
posteo    | janpublicebert
```


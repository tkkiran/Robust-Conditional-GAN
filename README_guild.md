# RCGAN and Guild AI

This project supports the following functionality with Guild AI by way
of [guild.yml](guild.yml):

- Track each training run as a separate experiment
- Compare run results
- Diff code changes and hyperparameters across runs
- View runs in TensorBoard

## Quick start

Below is a list of commands to perform common project tasks. Refer to
[Overview](#overview) and [Get started](#get-started) below to setup
your environment before running these commands.

| Dataset  | Model        | Command |
| -------- | ------------ | ------- |
| MNIST    | RCGAN        | `guild run mnist-rcgan:train` |
| MNIST    | RCGAN-U      | `guild run mnist-rcganu:train` |
| MNIST    | RCGAN+y      | `guild run mnist-rcgany:train` |
| MNIST    | Biased GAN   | `guild run mnist-biased:train` |
| MNIST    | Unbased GAN  | `guild run mnist-unbiased:train` |
| MNIST    | Ambient GAN  | `guild run mnist-ambient:train` |
| CIFAR-10 | RCGAN        | `guild run cifar10-rcgan: train` |
| CIFAR-10 | RCGAN-U      | `guild run cifar10-rcganu: train` |
| CIFAR-10 | Biased GAN   | `guild run cifar10-biased: train` |
| CIFAR-10 | Unbiased GAN | `guild run cifar10-unbiased: train` |

You can set hyperparameters for any runs using `guild run OPERATION
FLAG=VAL...` where `FLAG` is the name of the hyperparameter. To get
help for the project, including supported flags, run:

    $ guild help

To get help for a specific operation, use `guild run OPERATION
--help-op`. For example, to list available hyperparameters for
`train-cls` (classification task), run:

    $ guild run mnist-rcgan:train --help-op

## Overview

[Guild AI](https://guild.ai) is an open source command line tool that
automates project tasks. Guild AI works by reading configuration in
[guild.yml](guild.yml) - it does not require changes to project source
code. Guild AI is similar to tools like Maven or Grunt but with
features supporting machine learning workflow.

Below is a summary of Guild AI commands that can be used with this
project.

**`guild help`** <br> Show project help including models, operation,
and supported flags.

**`guild run [MODEL]:OPERATION [FLAG=VAL]...`** <br> Runs a model
operation. Runs are tracked as unique file system artifacts that can
be managed, inspected, and compared with other runs. Flags may be
specified to change operation behavior.

**`guild runs`** <br> List runs, including run ID, model and
operation, start time, status, and label.

**`guild runs rm RUN`** <br> Delete a run where `RUN` is a run ID or
listing index. You can delete multiple runs matching various criteria.

**`guild compare`** <br> Compare run results including loss and
validation accuracy.

**`guild tensorboard`** <br> View project runs in TensorBoard. You can
view all runs or runs matching various criteria.

**`guild diff RUN1 RUN2`** <br> Diff two runs. You can diff flags,
output, dependencies, and files using a variety of diff tools.

**`guild view`** <br> Open a web based run visualizer to compare and
inspect runs.

For a complete list of commands:

```
$ guild --help
```

For help with a specific command:

```
$ guild COMMAND --help
```

## Get started

The `guild` program is part of [Guild
AI](https://github.com/guildai/guildai) and can be installed using
pip.

Follow the steps below to install Guild AI and initialize a project
environment.

### Install Guild AI

To install Guild AI, use `pip`:

```
$ pip install guildai --upgrade
```

For additional information, see [Install Guild
AI](https://guild.ai/install/).

### Clone RCGAN repository

```
$ git clone https://github.com/POLane16/Robust-Conditional-GAN.git
```

### Initialize environment

Change to the project directory:

```
$ cd Robust-Conditional-GAN
```

Initialize an environment:

```
$ guild init
```

The `init` command creates a virtual environment in `env` and installs
Guild AI and the Python packages listed in
[`requirements.txt`](requirements.txt). Environments are used to
isolate project work from other areas of the system.

Activate the environment:

```
$ source guild-env
```

You can alternatively run `source env/bin/activate`, which is
equivalent to `source guild-env`.

Check the environment:

```
$ guild check
```

If you get errors, run `guild check --verbose` to get more information
and, if you can't resolve the issue, [open an
issue](https://github.com/guildai/guildai/issues) to get help.

## Train models on MNIST

The following models are trained on MNIST.

- `rcgan-mnist`
- `rcganu-mnist`
- `rcgany-mnist`
- `biased-mnist`
- `unbiased-mnist`
- `ambient-mnist`

To train a model, use the `train` operation as follows:

```
$ guild run MODEL:train
```

By default, models are trained over 100 epochs. You train over a
different number by setting the `epoch` flag:

```
$ guild run MODEL:train epoch=EPOCHS
```

Use `guild run MODEL:train --help-op` for a list of supported flags
for a particular train operation.

## View training progress in TensorBoard

To view training progress in TensorBoard, open a separate command
console.

In the new command console, change to the project directory:

```
$ cd Robust-Conditional-GAN
```

Activate the environment:

```
$ source guild-env
```

List project runs:

```
$ guild runs
```

Guild shows available runs, indicating their status (e.g. `running`,
`completed`, etc.)

To view the runs in TensorBoard, run:

```
$ guild tensorboard
```

If you run `guild tensorboard` on your workstation, Guild starts
TensorBoard on an available port and opens it in your browser. If you
run the command on a remote server, you have to open TensorBoard in
your browser manually. Use the link displayed in the console.

If you need to run TensorBoard on a specific port, use the `--port`
option:

```
$ guild tensorboard --port 8080
```

Guild automatically synchronizes TensorBoard with the current list of
run. You can leave TensorBoard running during your work.

## Train models on CIFAR-10

The following models are trained on CIFAR-10:

- `biased-cifar10`
- `rcgan-cifar10`
- `rcganu-cifar10`
- `unbiased-cifar10`

As with the MNIST models, you train a CIFAR-10 model using `guild run`
as follows:

```
$ guild run MODEL:train
```

You can control the length of training by specifying iterations. By
default, each CIFAR-10 model is trained over 50000 iterations. You can
specify a different number by specifying the `niters` flag:

```
$ guild run MODEL:train niters=ITERATIONS
```

## Compare model performance

You may compare model performance using TensorFlow (see steps above
for starting TensorFlow with Guild) or using the Guild AI `compare`
command.

To compare model loss and validation accuracy, run:

```
$ guild compare
```

Use the arrow keys to navigate within the Compare program.

Press `q` to exit the Compare program.

## Testing

This project supports a number of tests. To run the full test suite,
run:

```
$ guild test
```

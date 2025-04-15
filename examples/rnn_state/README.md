# RNN State Handling Example

This example demonstrates how to properly handle RNN state initialization and management in Haiku, specifically focusing on LSTM state handling.

## Overview

The example shows:
- How to properly initialize RNN/LSTM states
- How to manage state between forward passes
- How to handle batch dimensions in state
- Proper use of Haiku's transform API with RNN cores

## Files

- `rnn.py`: Contains the RNN implementation
- `train.py`: Shows how to use the RNN in practice
- `rnn_test.py`: Tests verifying the RNN functionality

## Running the Example

To run the training example:
```bash
python3 train.py
```

To run the tests:
```bash
python3 -m pytest rnn_test.py
```

## Expected Output

The training example will show:
- Input and output shapes
- Confirmation of successful forward pass

The tests verify:
- Correct output shapes for various batch sizes
- Proper state evolution between steps
- Correct state structure 
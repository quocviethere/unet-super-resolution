import matplotlib.pyplot as plt
import torch
import generate

def predict_and_display(model, test_dataloader, device):
    model.eval()

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(test_dataloader):
            if idx >= 10:
                break
            inputs = inputs.to(device)
            predictions = model(inputs)
            generate(model, inputs, labels)
            plt.show()


# Usage example:
# predict_and_display(model, test_dataloader, device)
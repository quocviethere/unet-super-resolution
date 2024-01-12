import matplotlib.pyplot as plt

def plot_result(filename, num_epochs, train_psnrs, eval_psnrs, train_losses, eval_losses):
    epochs = list(range(num_epochs))
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,6))

    # Plotting data
    axs[0].plot(epochs, train_psnrs, label="Training")
    axs[0].plot(epochs, eval_psnrs, label="Evaluation")
    axs[1].plot(epochs, train_losses, label="Training")
    axs[1].plot(epochs, eval_losses, label="Evaluation")

    # Setting labels for axes and adjusting their sizes
    axs[0].set_xlabel("Epochs", fontsize=18)
    axs[1].set_xlabel("Epochs", fontsize=18)
    axs[0].set_ylabel("PSNR", fontsize=18)
    axs[1].set_ylabel("Loss", fontsize=18)

    # Adjusting tick label sizes
    axs[0].tick_params(axis='both', which='major', labelsize=14)
    axs[1].tick_params(axis='both', which='major', labelsize=14)

    # Adjusting the legend size
    axs[0].legend(fontsize=14)
    axs[1].legend(fontsize=14)

    # Saving the plot
    plt.savefig(f'{filename}.pdf')
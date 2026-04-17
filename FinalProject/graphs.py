import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def get_score_arr(file_path, score="Accuracy", result="mean", plus_epoch=0):
    df = pd.read_csv(file_path)
    df = df.drop(columns=['epoch']).rename(columns={'Unnamed: 0': 'epoch'})
    score_arr = df[df["assessment"]==score][result].to_numpy()
    epoch_arr = df[df["assessment"]==score]["epoch"].to_numpy()
    epoch_arr = epoch_arr+plus_epoch

    return (1-score_arr), epoch_arr

def get_all_scores(string_start, score="Accuracy", result="mean"):
    err128, epc128 = get_score_arr(f"final_models/{string_start}_train128_val.csv",
                                   score=score, result=result, plus_epoch=0)
    err256, epc256 = get_score_arr(f"final_models/{string_start}_train256_val.csv",
                                   score=score, result=result, plus_epoch=50)
    err512, epc512 = get_score_arr(f"final_models/{string_start}_train512_val.csv",
                                   score=score, result=result, plus_epoch=100)

    err = np.concat([err128, err256, err512])
    epc = np.concat([epc128, epc256, epc512])

    return err, epc

def plot_PT(score="Accuracy", result="mean"):

    CE_err, CE_epc = get_all_scores("PT_CE", score=score, result=result)
    #DL_err, DL_epc = get_all_scores("PT_DL")
    LCDL_err, LCDL_epc = get_all_scores("PT_LCDL", score=score, result=result)

    plt.figure(figsize=(12, 5))

    # Plotting the main data
    plt.plot(CE_epc, CE_err, color='red', label='Cross Entropy Loss')
    #plt.plot(DL_epc, DL_err, color='green', label='Dice Loss')
    plt.plot(LCDL_epc, LCDL_err, color='blue', label='Log Cosh Dice Loss')

    # Adding vertical dashed (traced) lines
    plt.axvline(x=50, color='gray', linestyle='--')
    plt.axvline(x=100, color='gray', linestyle='--')

    # Adding region labels (adjusting y-position as needed)
    # Using text coordinates: (x, y, "text")
    y_text_pos = max(max(CE_err), max(LCDL_err)) * 0.95  # Places text near the top of the plot
    plt.text(15, y_text_pos, 'size = 128', fontsize=12, ha='center')
    plt.text(65, y_text_pos, '256', fontsize=12, ha='center')
    plt.text(115, y_text_pos, '512', fontsize=12, ha='center')

    # Formatting labels and legend
    plt.ylabel(f'1 - {result} {score}')
    plt.xlabel('Epoch')
    plt.legend()

    plt.show()


plot_PT("Recall")

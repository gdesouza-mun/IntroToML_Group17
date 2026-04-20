import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def get_results(file_path):
    df = pd.read_csv(file_path)
    df = df.drop(columns=['epoch']).rename(columns={'Unnamed: 0': 'epoch'})

    # 1. Filter the dataframe to only look at IoU rows
    iou_df = df[df['assessment'] == 'IoU']

    # 2. Find the index of the best mean IoU
    best_idx = iou_df['mean'].idxmax()

    # 3. Get the epoch number for that best row
    best_epoch = iou_df.loc[best_idx, 'epoch']

    # Show all columns, no matter how many
    pd.set_option('display.max_columns', None)

    # Show all rows (be careful if the DF is massive!)
    pd.set_option('display.max_rows', None)

    # Expand the width so long strings don't get cut off
    pd.set_option('display.width', 1000)

    print(df[df["epoch"]==best_epoch])


losses=["CE", "LCDL", "PT_CE", "PT_LCDL"]

for loss in losses:
    print(f"===================={loss}==================================")
    get_results(f"final_models/{loss}_train512_val.csv")
    print(f"============================================================")


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

def plot_mean(score="Accuracy", result="mean"):

    CE_err, CE_epc = get_all_scores("CE", score=score, result=result)
    LCDL_err, LCDL_epc = get_all_scores("LCDL", score=score, result=result)

    PT_CE_err, PT_CE_epc = get_all_scores("PT_CE", score=score, result=result)
    PT_LCDL_err, PT_LCDL_epc = get_all_scores("PT_LCDL", score=score, result=result)

    plt.figure(figsize=(12, 5))

    # Plotting the main data
    plt.plot(CE_epc, CE_err, color='red', label='Cross Entropy Loss')
    plt.plot(LCDL_epc, LCDL_err, color='blue', label='Log Cosh Dice Loss')

    plt.plot(PT_CE_epc, PT_CE_err, color='orange', label='Pre Trained Cross Entropy Loss')
    plt.plot(PT_LCDL_epc, PT_LCDL_err, color='purple', label='Pre Traind Log Cosh Dice Loss')

    ftsize1=18

    # Adding vertical dashed (traced) lines
    plt.axvline(x=50, color='gray', linestyle='--')
    plt.axvline(x=100, color='gray', linestyle='--')

    # Adding region labels (adjusting y-position as needed)
    # Using text coordinates: (x, y, "text")
    y_text_pos = max(max(CE_err), max(LCDL_err)) * 0.95  # Places text near the top of the plot
    plt.text(15, y_text_pos, 'size = 128 px', fontsize=ftsize1, ha='center')
    plt.text(65, y_text_pos, '256 px', fontsize=ftsize1, ha='center')
    plt.text(115, y_text_pos, '512 px', fontsize=ftsize1, ha='center')

    # Formatting labels and legend
    plt.ylabel(f'1 - Mean {score}', fontsize=ftsize1)
    plt.xlabel('Epoch', fontsize=ftsize1)
    plt.tick_params(axis='both', which='major', labelsize=ftsize1-4)
    plt.legend()

    plt.show()


def plt_by_class(loss, score="Recall"):

    pallet_nav={
        "soil" : [21, 171, 234/255.0],
        "bedrock" : [191, 21, 234],
        "sand" :  [234, 84, 21],
        "big rock" : [64, 234, 21],
        "null" :  [255,255, 255]
    }

    err_soil, epc_soil = get_all_scores(loss, score=score, result="soil")
    err_bedrock, epc_bedrock = get_all_scores(loss, score=score, result="bedrock")
    err_sand, epc_sand = get_all_scores(loss, score=score, result="sand")
    err_big_rock, epc_big_rock = get_all_scores(loss, score=score, result="big rock")
    err_mean, epc_mean = get_all_scores(loss, score=score, result="mean")
    ftsize1=18

    plt.plot(epc_soil, err_soil, color=np.array(pallet_nav["soil"])/255.0, label="soil")
    plt.plot(epc_bedrock, err_bedrock, color=np.array(pallet_nav["bedrock"])/255.0, label="bedrock")
    plt.plot(epc_sand, err_sand, color=np.array(pallet_nav["sand"])/255.0, label="sand")
    plt.plot(epc_big_rock, err_big_rock, color=np.array(pallet_nav["big rock"])/255.0, label="big rock")

    plt.plot(epc_mean, err_mean, color='darkgray', label="mean", linestyle = "--")

    # Adding vertical dashed (traced) lines
    plt.axvline(x=50, color='gray', linestyle='--')
    plt.axvline(x=100, color='gray', linestyle='--')

    y_text_pos = max(max(err_soil), max(err_bedrock),
                     max(err_sand), max(err_big_rock),
                     max(err_mean)) * 0.95  # Places text near the top of the plot
    plt.text(15, y_text_pos, 'size = 128 px', fontsize=ftsize1, ha='center')
    plt.text(65, y_text_pos, '256 px', fontsize=ftsize1, ha='center')
    plt.text(115, y_text_pos, '512 px', fontsize=ftsize1, ha='center')



    # Formatting labels and legend
    plt.ylabel(f'1 - {score}', fontsize=ftsize1)
    plt.xlabel('Epoch', fontsize=ftsize1)
    plt.tick_params(axis='both', which='major', labelsize=ftsize1-4)
    plt.legend(fontsize=ftsize1-4)

    if loss=="PT_CE":
        plt.title("Recall evolution with Cross Entropy loss (pre trained model)", fontsize=ftsize1)
    if loss=="PT_LCDL":
        plt.title("Recall evolution with Log Cosh Dice Loss (pre trained model)", fontsize=ftsize1)

    plt.show()




#plot_mean("IoU")
#plt_by_class("PT_LCDL")

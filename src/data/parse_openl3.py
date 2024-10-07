import os
import soundfile as sf
from tqdm import tqdm


import numpy as np
import torchopenl3

NPY_SAVE_MODE = False

def extract_openl3_feature(input_path, model=None, output_path = None):
    audio, sr = sf.read(input_path)
    if NPY_SAVE_MODE:
        model = torchopenl3.models.load_audio_embedding_model(input_repr="mel128", content_type="music", embedding_size=512)
        output, _ = torchopenl3.get_audio_embedding(audio, sr,  model=model)
        output = output.detach().cpu().numpy()
        np.save(output_path, output)
    else: 
        model = torchopenl3.models.load_audio_embedding_model(input_repr="mel128", content_type="music", embedding_size=512)
        output, _ = torchopenl3.get_audio_embedding(audio, sr, model=model)
        output = output.detach().cpu().numpy()
        return output

def extract_multiple_openl3_feature(input_dir, output_dir):

    input_file_list = os.listdir(input_dir)
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    model = torchopenl3.models.load_audio_embedding_model(input_repr="mel128", content_type="music", embedding_size=512)

    for input_file in tqdm(input_file_list):
        
        
        input_path = os.path.join(input_dir, input_file)

        output_file = input_file.replace('.wav', '.npy')
        output_path = os.path.join(output_dir, output_file)
        
        if input_file in [ "american_baroque-dances_and_suites_of_rameau_and_couperin-25-le_petit_rien_xiveme_ordre_couperin-88-117.wav", "norine_braun-now_and_zen-08-gently-117-146.wav", "jacob_heringman-josquin_des_prez_lute_settings-19-gintzler__pater_noster-204-233.mp3"]:
            continue
        if os.path.exists(output_path):
            continue

        extract_openl3_feature(input_path,  model, output_path)
        
def get_50_plus_multilabel_filepath():
    from helper import fifty_plus
    wav_paths = fifty_plus["mp3_path"].apply(convert_mp3_path_to_wav_path)
    return wav_paths.tolist()

def convert_mp3_path_to_wav_path(mp3_path):
    
    mp3_path = mp3_path.split("/")[-1]
    return mp3_path.replace(".mp3", ".wav")

def get_top50_filepath():
    from helper import get_n_class_k_shot_top50, get_eval_data_top50
    wav_paths = []

    for n in np.concatenate(([1],  np.arange(5, 51, 5))):
        n_way_val, tags_val = get_eval_data_top50(n, 2, "valid")
        n_way_test, tags_test = get_eval_data_top50(n, 2, "test")
        wav_paths.extend(n_way_val["mp3_path"].apply(convert_mp3_path_to_wav_path).tolist())
        wav_paths.extend(n_way_test["mp3_path"].apply(convert_mp3_path_to_wav_path).tolist())
        for k in np.concatenate(([1],  np.arange(5, 21, 5))):
            n_way_k_shot, tags = get_n_class_k_shot_top50(n, k)
            wav_paths.extend(n_way_k_shot["mp3_path"].apply(convert_mp3_path_to_wav_path).tolist())
    
    return list(set(wav_paths))

            
    
    

if __name__ == '__main__':

    # a = get_top50_filepath()
    import fire

    fire.Fire()
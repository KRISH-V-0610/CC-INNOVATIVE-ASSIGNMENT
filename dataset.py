import mne
from mne.datasets import eegbci
import numpy as np
import torch
from sklearn.model_selection import train_test_split

def load_and_prep_data():
    print("Downloading/Loading Advanced EEG Data (10 Subjects)...")
    print("This might take a few minutes depending on your internet speed.")
    
    subjects = range(1, 11) # Scaling up to 10 subjects
    runs = [4, 8, 12]       # Left/Right hand motor imagery
    
    all_X, all_y = [], []
    
    for subj in subjects:
        try:
            raw_fnames = eegbci.load_data(subjects=subj, runs=runs)
            raws = [mne.io.read_raw_edf(f, preload=True) for f in raw_fnames]
            raw = mne.io.concatenate_raws(raws)
            
            eegbci.standardize(raw)
            montage = mne.channels.make_standard_montage('standard_1005')
            raw.set_montage(montage)
            raw.filter(4., 30., fir_design='firwin', skip_by_annotation='edge', verbose=False)
            
            events, _ = mne.events_from_annotations(raw, event_id=dict(T1=2, T2=3), verbose=False)
            epochs = mne.Epochs(raw, events, event_id={'Left': 2, 'Right': 3}, 
                                tmin=-1., tmax=4., proj=True, baseline=None, preload=True, verbose=False)
            
            X = epochs.get_data(copy=True) * 1e6
            y = epochs.events[:, -1] - 2 # Map to 0 (Left) and 1 (Right)
            
            all_X.append(X)
            all_y.append(y)
        except Exception as e:
            print(f"Skipping subject {subj} due to error: {e}")
            
    # Combine all subjects into one massive dataset
    X_full = np.concatenate(all_X, axis=0)
    y_full = np.concatenate(all_y, axis=0)
    
    # Stratified split ensures equal distribution of Left/Right in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42, stratify=y_full)
    
    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    return X_train, X_test, y_train, y_test
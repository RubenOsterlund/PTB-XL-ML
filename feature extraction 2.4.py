import neurokit2 as nk
import numpy as np
import pandas as pd


def load_data(file_path):
    """Loads ECG data from specified npy file"""
    try:
        data = np.load(file_path)
        if data.ndim != 3 or data.shape[2] < 12:
            raise ValueError(f"Invalid data structure in {file_path}")
        return data  # Returns all patients (shape: [patients, time, leads])
    except Exception as e:
        print(f"Error loading data from {file_path}: {str(e)}")
        return None


def enhance_ecg(signal, sampling_rate=100):
    """Enhances ECG signal quality"""

    try:
            
        # Clean the signal
        cleaned = nk.ecg_clean(signal, sampling_rate=sampling_rate, method="neurokit")

        # Remove baseline wander
        baseline = nk.signal_smooth(cleaned, size=int(0.2 * sampling_rate))
        cleaned = cleaned - baseline

        # Apply bandpass filter
        return nk.signal_filter(cleaned, sampling_rate=sampling_rate, lowcut=0.5, highcut=40.0,
                                method="butterworth", order=3)
    except Exception as e:
        print(f"Signal processing error: {str(e)}")
        return None


def find_rpeaks(signal, sampling_rate=100):
    """Identifies R-peaks with fallback to Pan-Tompkins algorithm"""
    try:
        # First attempt with neurokit method
        _, rpeaks = nk.ecg_peaks(signal, sampling_rate=sampling_rate, method="neurokit")
        rpeaks = rpeaks["ECG_R_Peaks"]

        # If not enough R-peaks found, try Pan-Tompkins
        if len(rpeaks) < 0.5 * (len(signal) / sampling_rate * (60 / 80)):
            _, rpeaks = nk.ecg_peaks(signal, sampling_rate=sampling_rate, method="pantompkins")
            rpeaks = rpeaks["ECG_R_Peaks"]

        # Check that we have at least 2 R-peaks
        if len(rpeaks) < 2:
            print("Too few R-peaks detected")
            return np.array([])

        # Filter unrealistic R-peaks based on RR intervals
        rr_intervals = np.diff(rpeaks)
        median_rr = np.median(rr_intervals)

        # Start with first R-peak
        valid_peaks = [rpeaks[0]]

        # Add more R-peaks that meet criteria
        for i in range(1, len(rpeaks)):
            if 0.5 * median_rr < (rpeaks[i] - valid_peaks[-1]) < 1.5 * median_rr:
                valid_peaks.append(rpeaks[i])

        # If we still have too few peaks, return empty array
        if len(valid_peaks) < 2:
            return np.array([])

        return np.array(valid_peaks)
    except Exception as e:
        print(f"R-peak detection error: {str(e)}")
        return np.array([])


def delineate_waves(signal, rpeaks, sampling_rate=100):
    """Identifies P, Q, R, S, T waves and their boundaries"""
    waves = {
        'P_Peaks': [], 'P_Onsets': [], 'P_Offsets': [],
        'Q_Peaks': [], 'R_Peaks': rpeaks.tolist(), 'S_Peaks': [],
        'T_Peaks': [], 'T_Onsets': [], 'T_Offsets': [],
        'QRS_Onsets': [], 'QRS_Offsets': []
    }

    # If we don't have enough R-peaks, return empty results
    if len(rpeaks) < 2:
        return {k: np.array(v) for k, v in waves.items()}

    for i in range(len(rpeaks)):
        start = max(0, rpeaks[i] - int(0.2 * sampling_rate))
        end = min(len(signal), rpeaks[i] + int(0.4 * sampling_rate))

        # Ensure we have enough signal before and after the R peak
        if start == 0 or end >= len(signal) - 5:
            continue

        segment = signal[start:end]

        try:
            # Find Q wave - minimum before R peak
            q_window = signal[start:rpeaks[i]]
            if len(q_window) > 5:
                q_idx = start + np.argmin(q_window)
                waves['Q_Peaks'].append(q_idx)


                # Find QRS onset (start of QRS complex)
                qrs_onset_window = signal[max(0, q_idx - int(0.05 * sampling_rate)):q_idx]
                if len(qrs_onset_window) > 3:
                    # Find where the slope changes significantly before Q
                    qrs_onset = max(0, q_idx - int(0.05 * sampling_rate)) + np.argmin(
                        np.abs(np.diff(qrs_onset_window, prepend=qrs_onset_window[0])))
                    waves['QRS_Onsets'].append(qrs_onset)




            # Find S wave - minimum after R peak
            s_window = signal[rpeaks[i]:min(len(signal), rpeaks[i] + int(0.12 * sampling_rate))]
            if len(s_window) > 5:
                s_idx = rpeaks[i] + np.argmin(s_window)
                waves['S_Peaks'].append(s_idx)


                # Find QRS offset (end of QRS complex)
                qrs_offset_window = signal[s_idx:min(len(signal), s_idx + int(0.05 * sampling_rate))]
                if len(qrs_offset_window) > 3:
                    # Find where the slope changes significantly after S
                    qrs_offset = s_idx + np.argmin(np.abs(np.diff(qrs_offset_window, prepend=qrs_offset_window[0])))
                    waves['QRS_Offsets'].append(qrs_offset)


            # Find P wave - maximum in window before Q
            if 'Q_Peaks' in waves and len(waves['Q_Peaks']) > 0:
                last_q = waves['Q_Peaks'][-1]
                p_search_start = max(0, last_q - int(0.2 * sampling_rate))
                p_window = signal[p_search_start:last_q]
                if len(p_window) > 5:
                    p_peak = p_search_start + np.argmax(p_window)
                    waves['P_Peaks'].append(p_peak)
                    waves['P_Onsets'].append(max(0, p_peak - int(0.05 * sampling_rate)))
                    waves['P_Offsets'].append(min(last_q, p_peak + int(0.05 * sampling_rate)))

            # Find T wave - look for maximum after S wave within physiological limits
            if 'S_Peaks' in waves and len(waves['S_Peaks']) > 0:
                last_s = waves['S_Peaks'][-1]
                # T wave typically occurs within 160-300ms after R peak
                t_search_start = last_s
                t_search_end = min(len(signal), rpeaks[i] + int(0.4 * sampling_rate))
                t_window = signal[t_search_start:t_search_end]

                if len(t_window) > 10:
                    # Find maximum in the T-wave window
                    t_peak = t_search_start + np.argmax(t_window)

                    # Verify T wave is far enough from S wave to be physiologically valid
                    if t_peak - last_s > int(0.03 * sampling_rate):
                        waves['T_Peaks'].append(t_peak)
                        waves['T_Onsets'].append(max(last_s, t_peak - int(0.1 * sampling_rate)))
                        waves['T_Offsets'].append(min(len(signal) - 1, t_peak + int(0.1 * sampling_rate)))

        except Exception as e:
            # Silent error handling to continue to the next R-peak
            continue

    return {k: np.array(v) for k, v in waves.items()}


def measure_st_deviation(signal, waves, sampling_rate=100):
    """
    Measures ST segment deviation with improved reliability.
    Uses multiple methods to ensure ST measurement even when QRS delineation is imperfect.
    """
    st_deviations = []

    # Method 1: Use R peaks directly if available (most reliable reference point)
    if len(waves['R_Peaks']) >= 2:
        for r_peak in waves['R_Peaks']:
            # Make sure we have enough signal after the R peak
            if r_peak + int(0.12 * sampling_rate) >= len(signal):
                continue

            # Measure at fixed point after R peak (120ms is typically in ST segment)
            st_point = r_peak + int(0.12 * sampling_rate)

            # Use PR segment as baseline (80-20ms before R peak)
            pr_start = max(0, r_peak - int(0.08 * sampling_rate))
            pr_end = max(0, r_peak - int(0.02 * sampling_rate))

            if pr_start < pr_end and pr_end - pr_start >= 3:  # Need at least 3 points for meaningful average
                baseline = np.median(signal[pr_start:pr_end])  # Median is more robust to outliers
                st_value = signal[st_point]
                st_deviation = st_value - baseline
                st_deviations.append(st_deviation)

    # Method 2: Try using S peaks and QRS offsets if available and Method 1 failed
    if len(st_deviations) == 0 and len(waves['S_Peaks']) >= 2:
        for s_peak in waves['S_Peaks']:
            # Measure 80ms after S peak
            st_point = min(len(signal) - 1, s_peak + int(0.08 * sampling_rate))

            # Find closest R peak before this S peak
            r_peaks_before = waves['R_Peaks'][waves['R_Peaks'] < s_peak]
            if len(r_peaks_before) == 0:
                continue

            r_peak = r_peaks_before[-1]

            # Use PR segment as baseline
            pr_start = max(0, r_peak - int(0.08 * sampling_rate))
            pr_end = max(0, r_peak - int(0.02 * sampling_rate))

            if pr_start < pr_end and pr_end - pr_start >= 3:
                baseline = np.median(signal[pr_start:pr_end])
                st_deviation = signal[st_point] - baseline
                st_deviations.append(st_deviation)

    # Method 3: Fallback method if other methods failed
    # Use simple amplitude-based approach
    if len(st_deviations) == 0 and len(waves['R_Peaks']) >= 2:
        # Find average beat length
        rr_intervals = np.diff(waves['R_Peaks'])
        if len(rr_intervals) > 0:
            avg_rr = np.median(rr_intervals)

            for r_peak in waves['R_Peaks']:
                # Skip if we don't have enough signal after R peak
                if r_peak + int(0.3 * avg_rr) >= len(signal):
                    continue

                # ST point at ~30% of RR interval (typically in ST segment)
                st_point = r_peak + int(0.3 * avg_rr)

                # Use signal at fixed percentage before R as baseline
                baseline_point = max(0, r_peak - int(0.1 * avg_rr))

                # Simple difference between ST point and baseline
                st_deviation = signal[st_point] - signal[baseline_point]
                st_deviations.append(st_deviation)

    # Return the mean ST deviation if we have measurements
    if st_deviations:
        # Filter out extreme values (beyond physiological range)
        filtered_deviations = [d for d in st_deviations if abs(d) < 0.5]  # Assuming normalized signal
        if filtered_deviations:
            return np.mean(filtered_deviations)

    # If we reach here, we couldn't measure ST deviation reliably
    return np.nan


def check_q_wave(signal, waves, sampling_rate=100):
    """Checks for pathological Q waves (>25% of R amplitude or >40ms duration)"""
    if len(waves['Q_Peaks']) == 0 or len(waves['R_Peaks']) == 0:
        return 0, np.nan, np.nan  # No Q waves detected

    pathological_count = 0
    q_amplitudes = []
    q_durations = []

    for i, q_peak in enumerate(waves['Q_Peaks']):
        # Find closest R peak after this Q
        r_peaks_after = waves['R_Peaks'][waves['R_Peaks'] > q_peak]
        if len(r_peaks_after) == 0:
            continue

        r_peak = r_peaks_after[0]

        # Q amplitude (negative)
        q_amplitude = abs(signal[q_peak])
        r_amplitude = abs(signal[r_peak])

        # Q duration
        # Find QRS onset before this Q
        qrs_onsets_before = waves['QRS_Onsets'][waves['QRS_Onsets'] < q_peak]
        if len(qrs_onsets_before) == 0:
            # If no explicit onset found, estimate
            qrs_onset = max(0, q_peak - int(0.04 * sampling_rate))
        else:
            qrs_onset = qrs_onsets_before[-1]

        q_duration = (q_peak - qrs_onset) / sampling_rate * 1000  # in ms

        # Check if pathological
        if q_amplitude > 0.25 * r_amplitude or q_duration > 40:
            pathological_count += 1

        q_amplitudes.append(q_amplitude)
        q_durations.append(q_duration)

    # Return pathological flag, mean amplitude and duration
    has_pathological_q = 1 if pathological_count > 0 else 0
    mean_q_amplitude = np.mean(q_amplitudes) if q_amplitudes else np.nan
    mean_q_duration = np.mean(q_durations) if q_durations else np.nan

    return has_pathological_q, mean_q_amplitude, mean_q_duration


def analyze_t_wave(signal, waves):
    """Analyzes T wave characteristics (amplitude, inversion)"""
    if len(waves['T_Peaks']) == 0:
        return 0, np.nan  # No T waves detected

    t_amplitudes = []
    for t_peak in waves['T_Peaks']:
        if t_peak < len(signal):
            t_amplitudes.append(signal[t_peak])

    if not t_amplitudes:
        return 0, np.nan

    # T wave inversion if amplitude is negative
    is_inverted = 1 if np.mean(t_amplitudes) < 0 else 0
    mean_amplitude = np.mean(t_amplitudes)

    return is_inverted, mean_amplitude


def calculate_rs_ratio(signal, waves):
    """Calculates R/S amplitude ratio"""
    if len(waves['R_Peaks']) == 0 or len(waves['S_Peaks']) == 0:
        return np.nan

    # Match R peaks with S peaks
    rs_ratios = []
    for r_peak in waves['R_Peaks']:
        # Find the next S peak after this R peak
        s_peaks_after = waves['S_Peaks'][waves['S_Peaks'] > r_peak]
        if len(s_peaks_after) > 0:
            s_peak = s_peaks_after[0]

            # Calculate absolute amplitudes
            r_amplitude = abs(signal[r_peak])
            s_amplitude = abs(signal[s_peak])

            # Calculate ratio (avoid division by zero)
            if s_amplitude > 0:
                rs_ratios.append(r_amplitude / s_amplitude)

    return np.mean(rs_ratios) if rs_ratios else np.nan


def measure_r_wave(signal, waves):
    """Measures R wave amplitude and duration"""
    if len(waves['R_Peaks']) == 0:
        return np.nan, np.nan

    r_amplitudes = []
    r_durations = []

    for r_peak in waves['R_Peaks']:
        # Amplitude
        r_amplitude = signal[r_peak]
        r_amplitudes.append(r_amplitude)

        # Duration - find points where R wave starts and ends
        # Find Q peak before R
        q_peaks_before = waves['Q_Peaks'][waves['Q_Peaks'] < r_peak]
        # Find S peak after R
        s_peaks_after = waves['S_Peaks'][waves['S_Peaks'] > r_peak]

        if len(q_peaks_before) > 0 and len(s_peaks_after) > 0:
            q_peak = q_peaks_before[-1]
            s_peak = s_peaks_after[0]

            # Find where signal crosses zero or reaches minimum between Q and R
            segment_qr = signal[q_peak:r_peak]
            if len(segment_qr) > 3:
                # Look for where signal crosses from negative to positive
                zero_crossings = np.where(np.diff(np.signbit(segment_qr)))[0]
                if len(zero_crossings) > 0:
                    r_start = q_peak + zero_crossings[-1]
                else:
                    # If no zero crossing, use minimum point
                    r_start = q_peak + np.argmin(segment_qr)

                # Find where signal crosses zero or reaches minimum between R and S
                segment_rs = signal[r_peak:s_peak]
                if len(segment_rs) > 3:
                    # Look for where signal crosses from positive to negative
                    zero_crossings = np.where(np.diff(np.signbit(segment_rs)))[0]
                    if len(zero_crossings) > 0:
                        r_end = r_peak + zero_crossings[0]
                    else:
                        # If no zero crossing, use minimum point
                        r_end = r_peak + np.argmin(segment_rs)

                    # Calculate R duration in ms
                    r_duration = (r_end - r_start) / 100 * 1000  # assuming 100Hz sampling rate
                    r_durations.append(r_duration)

    mean_r_amplitude = np.mean(r_amplitudes) if r_amplitudes else np.nan
    mean_r_duration = np.mean(r_durations) if r_durations else np.nan

    return mean_r_amplitude, mean_r_duration


def measure_qrs_complex(signal, waves, sampling_rate=100):
    """Measures QRS complex characteristics"""
    if len(waves['QRS_Onsets']) < 2 or len(waves['QRS_Offsets']) < 2:
        return np.nan, np.nan

    # Make sure we have matching onsets and offsets
    min_len = min(len(waves['QRS_Onsets']), len(waves['QRS_Offsets']))

    qrs_durations = []
    qrs_areas = []

    for i in range(min_len):
        onset = waves['QRS_Onsets'][i]
        offset = waves['QRS_Offsets'][i]

        if offset > onset:
            # Duration in ms
            duration = (offset - onset) / sampling_rate * 1000
            qrs_durations.append(duration)

            # Area under curve (absolute)
            qrs_segment = signal[onset:offset]
            area = np.trapezoid(np.abs(qrs_segment))
            qrs_areas.append(area)

    mean_qrs_duration = np.mean(qrs_durations) if qrs_durations else np.nan
    mean_qrs_area = np.mean(qrs_areas) if qrs_areas else np.nan

    return mean_qrs_duration, mean_qrs_area


def extract_lead_features(signal, lead_name, sampling_rate=100):
    """Extracts specific features based on the lead"""
    # Process signal
    processed = enhance_ecg(signal, sampling_rate)
    if processed is None:
        return {}

    # Detect R-peaks
    rpeaks = find_rpeaks(processed, sampling_rate)
    if len(rpeaks) < 2:
        return {}

    # Delineate all waves
    waves = delineate_waves(processed, rpeaks, sampling_rate)

    # Common features for all leads
    features = {}

    # ST deviations
    features[f"{lead_name}_st_deviation"] = measure_st_deviation(processed, waves, sampling_rate)

    # Q wave analysis (for II, III, aVF primarily)
    has_pathological_q, q_amplitude, q_duration = check_q_wave(processed, waves, sampling_rate)
    features[f"{lead_name}_pathological_q"] = has_pathological_q
    features[f"{lead_name}_q_amplitude"] = q_amplitude
    features[f"{lead_name}_q_duration"] = q_duration

    # T wave analysis
    t_inverted, t_amplitude = analyze_t_wave(processed, waves)
    features[f"{lead_name}_t_inverted"] = t_inverted
    features[f"{lead_name}_t_amplitude"] = t_amplitude

    # For V1-V3, additional features
    if lead_name in ['V1', 'V2', 'V3']:
        # R wave measurements
        r_amplitude, r_duration = measure_r_wave(processed, waves)
        features[f"{lead_name}_r_amplitude"] = r_amplitude
        features[f"{lead_name}_r_duration"] = r_duration

        # QRS complex measurements
        qrs_duration, qrs_area = measure_qrs_complex(processed, waves, sampling_rate)
        features[f"{lead_name}_qrs_duration"] = qrs_duration
        features[f"{lead_name}_qrs_area"] = qrs_area

    # For V1-V2, R/S ratio
    if lead_name in ['V1', 'V2']:
        features[f"{lead_name}_rs_ratio"] = calculate_rs_ratio(processed, waves)

    return features


def process_patient(patient_data, patient_id, sampling_rate=100):
    """Processes all leads for a patient, focusing on specified leads and features"""
    # Define lead indices and names
    lead_indices = {
        'I': 0, 'II': 1, 'III': 2, 'aVR': 3, 'aVL': 4, 'aVF': 5,
        'V1': 6, 'V2': 7, 'V3': 8, 'V4': 9, 'V5': 10, 'V6': 11
    }

    # Define leads of interest for specific features
    inferior_leads = ['II', 'III', 'aVF']
    anterior_leads = ['V1', 'V2', 'V3']

    # Initialize ALL possible features with NaN values
    all_possible_features = {
        'patient_id': patient_id,
        # Derived features
        'inferior_st_elevation': np.nan,
        'anterior_st_elevation': np.nan,
        'inferior_q_waves': 0,  # Initialize binary flags to 0
        'anterior_t_inversion': 0
    }

    # Initialize all individual lead features with NaN or 0 for binary flags
    for lead in inferior_leads + anterior_leads:
        all_possible_features[f"{lead}_st_deviation"] = np.nan
        all_possible_features[f"{lead}_pathological_q"] = 0  # Binary flag
        all_possible_features[f"{lead}_q_amplitude"] = np.nan
        all_possible_features[f"{lead}_q_duration"] = np.nan
        all_possible_features[f"{lead}_t_inverted"] = 0  # Binary flag
        all_possible_features[f"{lead}_t_amplitude"] = np.nan

        # Special features for anterior leads
        if lead in anterior_leads:
            all_possible_features[f"{lead}_r_amplitude"] = np.nan
            all_possible_features[f"{lead}_r_duration"] = np.nan
            all_possible_features[f"{lead}_qrs_duration"] = np.nan
            all_possible_features[f"{lead}_qrs_area"] = np.nan

        # R/S ratio only for V1, V2
        if lead in ['V1', 'V2']:
            all_possible_features[f"{lead}_rs_ratio"] = np.nan

    # Process inferior leads (II, III, aVF)
    for lead_name in inferior_leads:
        lead_idx = lead_indices[lead_name]
        if lead_idx < patient_data.shape[1]:
            lead_signal = patient_data[:, lead_idx]

            # Extract lead-specific features
            lead_features = extract_lead_features(lead_signal, lead_name, sampling_rate)
            # Update all_possible_features with extracted values (if any)
            for key, value in lead_features.items():
                all_possible_features[key] = value

    # Process anterior leads (V1-V3)
    for lead_name in anterior_leads:
        lead_idx = lead_indices[lead_name]
        if lead_idx < patient_data.shape[1]:
            lead_signal = patient_data[:, lead_idx]

            # Extract lead-specific features
            lead_features = extract_lead_features(lead_signal, lead_name, sampling_rate)
            # Update all_possible_features with extracted values (if any)
            for key, value in lead_features.items():
                all_possible_features[key] = value

    # Compute derived features

    # Inferior ST elevation (average of II, III, aVF)
    st_inferior = []
    for lead in inferior_leads:
        if not np.isnan(all_possible_features[f"{lead}_st_deviation"]):
            st_inferior.append(all_possible_features[f"{lead}_st_deviation"])
    if st_inferior:
        all_possible_features["inferior_st_elevation"] = np.mean(st_inferior)

    # Anterior ST elevation (average of V1-V3)
    st_anterior = []
    for lead in anterior_leads:
        if not np.isnan(all_possible_features[f"{lead}_st_deviation"]):
            st_anterior.append(all_possible_features[f"{lead}_st_deviation"])
    if st_anterior:
        all_possible_features["anterior_st_elevation"] = np.mean(st_anterior)

    # Inferior Q wave presence (any in II, III, aVF)
    inferior_q = 0
    for lead in inferior_leads:
        if all_possible_features[f"{lead}_pathological_q"] == 1:
            inferior_q = 1
            break
    all_possible_features["inferior_q_waves"] = inferior_q

    # T wave inversions in anterior leads
    anterior_t_inversion = 0
    for lead in anterior_leads:
        if all_possible_features[f"{lead}_t_inverted"] == 1:
            anterior_t_inversion = 1
            break
    all_possible_features["anterior_t_inversion"] = anterior_t_inversion

    return pd.DataFrame([all_possible_features])


def process_dataset(data_file, output_file, dataset_name):
    """Process a single dataset and save results to a CSV file"""
    # Load data
    dataset = load_data(data_file)
    if dataset is None:
        print(f"Failed to load {data_file}")
        return False

    # Process all patients
    all_features = []
    num_patients = dataset.shape[0]

    print(f"Processing {dataset_name} dataset: {num_patients} patients")

    for i in range(num_patients):
        patient_data = dataset[i]
        patient_id = f"{dataset_name}_patient_{i + 1}"
        if i % 10 == 0:
            print(f"Processing {patient_id}...")

        patient_features = process_patient(patient_data, patient_id)
        if not patient_features.empty:
            all_features.append(patient_features)
            if i % 10 == 0 and i > 0:
                print(f"Progress: {i}/{num_patients} patients processed")

    # Compile and save
    if all_features:
        result_df = pd.concat(all_features, ignore_index=True)
        result_df.to_csv(output_file, index=False)
        print(f"\nFeatures saved for {len(all_features)} patients to '{output_file}'")

        # Show statistics
        print(f"Statistics of extracted features for {dataset_name}:")
        print(f"Total number of rows: {len(result_df)}")
        print(f"Number of patients: {result_df['patient_id'].nunique()}")

        # Group features by type
        feature_groups = {
            "ST changes": [col for col in result_df.columns if "st_" in col],
            "Q wave features": [col for col in result_df.columns if "_q_" in col or "_pathological_q" in col],
            "T wave features": [col for col in result_df.columns if "_t_" in col],
            "R wave features": [col for col in result_df.columns if "_r_" in col],
            "QRS features": [col for col in result_df.columns if "qrs_" in col],
            "R/S ratio": [col for col in result_df.columns if "rs_ratio" in col]
        }

        # Print missing values by feature group
        print("\nMissing values percentage by feature group:")
        for group, cols in feature_groups.items():
            if cols:
                miss_pct = result_df[cols].isna().mean().mean() * 100
                print(f"  {group}: {miss_pct:.1f}%")

        return True
    else:
        print(f"No features could be extracted for {dataset_name} dataset.")
        return False


def main():
    # Define datasets to process
    datasets = [
        {"file": "X_train_1.npy", "output": "ecg_cardiac_features_train.csv", "name": "train"},
        {"file": "X_valid_1.npy", "output": "ecg_cardiac_features_val.csv", "name": "val"},
        {"file": "X_test_1.npy", "output": "ecg_cardiac_features_test.csv", "name": "test"}
    ]

    # Process each dataset
    successful_datasets = 0
    for dataset in datasets:
        print(f"\n{'=' * 50}")
        print(f"Processing {dataset['name']} dataset from {dataset['file']}")
        print(f"{'=' * 50}")

        success = process_dataset(dataset['file'], dataset['output'], dataset['name'])
        if success:
            successful_datasets += 1

    print(f"\nProcessing complete. Successfully processed {successful_datasets}/{len(datasets)} datasets.")


if __name__ == "__main__":
    main()
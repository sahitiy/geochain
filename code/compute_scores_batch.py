import torch
import os
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast 
import numpy as np
import json

cls = np.array(['wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road, route', 'bed', 'window ', 'grass', 'cabinet', 'sidewalk, pavement', 'person', 'earth, ground', 'door', 'table', 'mountain, mount', 'plant', 'curtain', 'chair', 'car', 'water', 'painting, picture', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair', 'seat', 'fence', 'desk', 'rock, stone', 'wardrobe, closet, press', 'lamp', 'tub', 'rail', 'cushion', 'base, pedestal, stand', 'box', 'column, pillar', 'signboard, sign', 'chest of drawers, chest, bureau, dresser', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator, icebox', 'grandstand, covered stand', 'path', 'stairs', 'runway', 'case, display case, showcase, vitrine', 'pool table, billiard table, snooker table', 'pillow', 'screen door, screen', 'stairway, staircase', 'river', 'bridge, span', 'bookcase', 'blind, screen', 'coffee table', 'toilet, can, commode, crapper, pot, potty, stool, throne', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove', 'palm, palm tree', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar', 'arcade machine', 'hovel, hut, hutch, shack, shanty', 'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning, sunshade, sunblind', 'street lamp', 'booth', 'tv', 'plane', 'dirt track', 'clothes', 'pole', 'land, ground, soil', 'bannister, banister, balustrade, balusters, handrail', 'escalator, moving staircase, moving stairway', 'ottoman, pouf, pouffe, puff, hassock', 'bottle', 'buffet, counter, sideboard', 'poster, posting, placard, notice, bill, card', 'stage', 'van', 'ship', 'fountain', 'conveyer belt, conveyor belt, conveyer, conveyor, transporter', 'canopy', 'washer, automatic washer, washing machine', 'plaything, toy', 'pool', 'stool', 'barrel, cask', 'basket, handbasket', 'falls', 'tent', 'bag', 'minibike, motorbike', 'cradle', 'oven', 'ball', 'food, solid food', 'step, stair', 'tank, storage tank', 'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket, cover', 'sculpture', 'hood, exhaust hood', 'sconce', 'vase', 'traffic light', 'tray', 'trash can', 'fan', 'pier', 'crt screen', 'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass, drinking glass', 'clock', 'flag'])
weights = [0.10875312598137159, 0.22782079363438223, 0.1682270176863706, 0.03810142076260751, 0.1958401109625654, 0.063554588279477, 1.0, 0.00429082773900915, 0.05069200866324592, 0.14859857183885966, 0.0018115759015403842, 0.453037029916481, 0.011059382162204727, 0.2533939581032943, 0.06404497243439089, 0.042958094738197394, 0.21159702593205587, 0.10726212937337766, 0.023216003275786806, 0.018854909268714845, 0.2647252740954065, 0.012137799263347567, 0.029763077991828132, 0.022686646060857015, 0.0761703203252065, 0.17557256999885518, 0.12942190093060854, 0.07240050312851457, 0.06782959210532469, 0.1534834576078631, 0.0, 0.054876561861683514, 0.253214900224246, 0.0008220616261612096, 0.19880643756723027, 0.0008376794521822793, 0.0543922996160292, 0.013266801563102993, 0.3174680711481328, 0.01134332186720067, 0.03401865545094766, 0.037098356754591076, 0.5666560402491322, 0.4912532420191637, 0.002784145262842802, 0.023827060092071105, 0.14536978364077033, 0.0060792856920914495, 0.0735831635395584, 0.028071999372523596, 0.01272337983429472, 0.05409554560551889, 0.5432143243075093, 0.06757881295313528, 0.13825301878878213, 0.00513517531935447, 0.002546755581195223, 0.008656712361718298, 0.0061613218043033635, 0.061007892791440754, 0.15960305000013955, 0.07339974123507612, 0.0033827542626461474, 0.022296556842504527, 0.01208320004536453, 0.030091261188987722, 0.026449096395342517, 0.001782064190213641, 0.22299886175471612, 0.007244163001488707, 0.026276413254588953, 0.06882556342917678, 0.23003064990337416, 0.04682281686855221, 0.004429102874850328, 0.0, 0.033976769409549606, 0.02143219282187274, 0.011140738233700185, 0.19060370749178462, 0.11191832661295731, 0.01924153603082055, 0.06402394739197584, 0.10951620878444565, 0.07935803083834335, 0.0331838428178853, 0.01632238472628823, 0.47558048375570283, 0.05991110380939457, 0.005028249510336053, 0.11978693429866023, 0.2393755817510923, 0.016981999023110483, 0.33100957537712283, 0.42445512056508605, 0.3136666511857401, 0.0061071587906564246, 0.09338190051111724, 0.03518506966777248, 0.008785956983577821, 0.19779784311064447, 0.002575609577768679, 0.10005150512908928, 0.024589809129341857, 0.02261563631080785, 0.02282552044811365, 0.04628628275703596, 0.0009522641611995059, 0.005245671919354264, 0.0018385622781357195, 0.016537567329500145, 0.007210991360720969, 0.04858501824554905, 0.050641752123032756, 0.04360961941009033, 0.00600524742448132, 0.07990074200581138, 0.008437486529831428, 0.009777204224537886, 0.011413106062842158, 0.06082053189474603, 0.06484967833984664, 0.01237275342857479, 0.10819204038562762, 0.012853954523868675, 0.007048755493695816, 0.014888264915279732, 0.07312487387625713, 0.04091875647061467, 0.003433634039895123, 0.10550252306858474, 0.06223865480799576, 0.005130128198171204, 0.010732656352311304, 0.015327819485456869, 0.022979688275253343, 0.4858943242308143, 0.008814551932843374, 0.014696333855211893, 0.013234356730673484, 0.056940709527395855, 0.033338638217252685, 0.2027964188814377, 0.024370031823618913, 0.023261076986754142, 0.003675704388219124, 0.02265551897622887, 0.006998894740670512, 0.013797006932313527, 0.28045355244417924]


def custom_collate_fn(batch):
    valid_items = [item for item in batch if item[0] is not None]

    if not valid_items:
        return None, None, None # No valid images in this batch

    pil_images = [item[0] for item in valid_items]
    original_sizes = [item[1] for item in valid_items]
    image_paths = [item[2] for item in valid_items] # For reference and error reporting

    return pil_images, original_sizes, image_paths


def numerical_sort_key(string_path):
    try:
        filename = os.path.basename(string_path)
        file_number_str = os.path.splitext(filename)[0]
        file_number = int(file_number_str)
        return file_number
    except (AttributeError, IndexError, ValueError): 
        print(f"Warning: Could not parse number from filename '{string_path}' for sorting. Using raw path.")
        return string_path


def compute_scores(model, processor, dataset):
   
    BATCH_SIZE = 20  
    NUM_WORKERS = 2  
    USE_AMP = True   

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if USE_AMP and device.type != 'cuda':
        print("Warning: USE_AMP is True, but CUDA is not available. AMP will not be used.")
        USE_AMP = False

    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device) # Stays on CPU

    loc_values_map = {} 
    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False, 
        num_workers=NUM_WORKERS,
        collate_fn=custom_collate_fn,
        pin_memory=True if device.type == 'cuda' else False # Can speed up CPU to GPU transfers
    )

    print("Starting batch processing...")
    for batch_data in tqdm(data_loader, desc="Processing Batches"):
        if batch_data[0] is None: 
            print("Skipping an empty or fully problematic batch from DataLoader.")
            continue

        batch_pil_images, batch_original_sizes, batch_image_paths = batch_data

        if not batch_pil_images: 
            continue

        try:
            
            inputs = processor(images=batch_pil_images, return_tensors="pt")
            inputs_on_device = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                with autocast(enabled=(USE_AMP and device.type == 'cuda')):
                    outputs = model(**inputs_on_device)
           
            predicted_semantic_maps_batch = processor.post_process_semantic_segmentation(
                outputs,
                target_sizes=batch_original_sizes
            )

            for i, semantic_map_tensor in enumerate(predicted_semantic_maps_batch):
                current_image_path = batch_image_paths[i]
               
                if not isinstance(semantic_map_tensor, torch.Tensor):
                    print(f"Warning: Expected a Tensor for semantic map of {current_image_path}, got {type(semantic_map_tensor)}. Skipping locatability calculation.")
                    loc_values_map[current_image_path] = float('nan') # Or some error indicator
                    continue

                total_pixels = semantic_map_tensor.numel()
                loc_value = 0.0

                if total_pixels > 0:
                    unique_classes, counts = torch.unique(semantic_map_tensor, return_counts=True)
                    valid_mask = unique_classes < len(weights_tensor)
                    valid_classes = unique_classes[valid_mask]
                    valid_counts = counts[valid_mask]

                    if not torch.all(valid_mask): 
                        invalid_classes = unique_classes[~valid_mask]
                        print(f"Warning: For image {current_image_path}, class IDs {invalid_classes.tolist()} are out of bounds "
                              f"(max index: {len(weights_tensor)-1}). Skipping these classes.")

                    if valid_classes.numel() > 0:
                        class_specific_weights = weights_tensor[valid_classes]
                        class_indices = valid_classes.cpu().numpy()
                        class_names = cls[class_indices]
                        percentages = valid_counts.float() / total_pixels
                        class_map_dict = dict(zip(class_names, percentages.cpu().numpy()))
                        loc_value = torch.sum(class_specific_weights * percentages).item()
               
                loc_values_map[current_image_path] = (loc_value, class_map_dict)

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"CUDA out of memory for batch starting with {batch_image_paths[0] if batch_image_paths else 'N/A'} "
                      f"(batch size: {len(batch_pil_images)}). Try reducing BATCH_SIZE. Skipping this batch.")
            else:
                print(f"A runtime error occurred for batch starting with {batch_image_paths[0] if batch_image_paths else 'N/A'}: {e}. Skipping this batch.")
            for path in batch_image_paths:
                if path not in loc_values_map: loc_values_map[path] = float('nan')
        except Exception as e:
            print(f"An unexpected error occurred for batch starting with {batch_image_paths[0] if batch_image_paths else 'N/A'}: {e}. Skipping this batch.")
            for path in batch_image_paths:
                if path not in loc_values_map: loc_values_map[path] = float('nan')

    final_scores = [loc_values_map[p][0] for p in loc_values_map]
    final_mapping = []
    for p in loc_values_map:
        map_dict = {}
        for k, v in loc_values_map[p][1].items():
            map_dict[str(k)] = float(v)
        final_mapping.append(json.dumps(map_dict))

    df = pd.DataFrame({
        'key': list(loc_values_map.keys()),
        'locatability_scores': final_scores,
        'class_mapping': final_mapping
    })
   
    return df
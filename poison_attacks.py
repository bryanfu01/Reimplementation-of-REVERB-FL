import torch
import torch.nn.functional as F

def fgsm_attack(model, original_audio, labels, device, epsilon = 0.02):
    """
    Applies the Fast Gradient Sign Method (FGSM) to audio STFT data.
    Note that data should be unpacked into the audio_data and the labels as input
    """
    # 1. Clone the audio and tell PyTorch to track gradients for it
    audio_tensor = original_audio.clone().detach().to(device)
    audio_tensor.requires_grad_(True)

    # 2. Dummy forward pass to evaluate the current loss landscape
    scores = model(audio_tensor)
    loss = F.cross_entropy(scores, labels)

    # 3. Dummy backward pass to populate audio_tensor.grad
    model.zero_grad()
    loss.backward()

    # 4. Collect the element-wise sign of the data gradient
    data_grad = audio_tensor.grad
    sign_data_grad = data_grad.sign()
    
    # 5. Create the perturbed audio safely without tracking gradients
    with torch.no_grad():
        perturbed_audio = audio_tensor + epsilon * sign_data_grad
        
        # 6. Dynamically clip to the valid domain X of the original audio
        domain_min = torch.min(original_audio)
        domain_max = torch.max(original_audio)
        perturbed_audio = torch.clamp(perturbed_audio, min=domain_min, max=domain_max)
    
    return perturbed_audio.detach()

def pgd_attack(model, original_audio, labels, device, epsilon = 0.02, num_iter = 50):
    """
    Projected Gradient Descent (PGD) matching Equation 8.
    Note that data should be unpacked into the audio_data and the labels as input
    """
    # 1. Random Initialization: X_tilde^(0) = X + u 
    # Sample uniform noise 'u' between -epsilon and epsilon
    u = torch.empty_like(original_audio).uniform_(-epsilon, epsilon).to(device)
    perturbed_audio = original_audio + u
    
    # 2. Define the fixed walls of the hypercube Pi_B_eps(X)
    lower_bound = original_audio - epsilon
    upper_bound = original_audio + epsilon
    
    # Force the initial random jump to stay inside the box
    perturbed_audio = torch.max(torch.min(perturbed_audio, upper_bound), lower_bound)
    
    # 3. Step size = epsilon / I
    alpha = epsilon / num_iter
    
    for i in range(num_iter):
        # We must tell PyTorch to track gradients for the CURRENT perturbed position
        perturbed_audio = perturbed_audio.clone().detach().requires_grad_(True)
        
        scores = model(perturbed_audio)
        loss = F.cross_entropy(scores, labels)
        
        model.zero_grad()
        loss.backward()
        
        # --- THE UPDATE STEP ---
        with torch.no_grad():
            # X_tilde^(i) + (epsilon/I) * sign(grad)
            step = alpha * perturbed_audio.grad.sign()
            perturbed_audio = perturbed_audio + step
            
            # --- THE PROJECTION STEP ---
            # Project back onto the L_inf ball centered at the ORIGINAL audio
            perturbed_audio = torch.max(torch.min(perturbed_audio, upper_bound), lower_bound)

    # Return the final adversarial tensor, detached from the computation graph
    return perturbed_audio.detach()

def awgn_attack(original_audio, device = None, std_dev = 0.03):
    """
    Adds gaussian noise to original data, should be called in constructor of client
    
    :param original_audio: raw audio data
    """
    with torch.no_grad():
        # Creates gaussian random variable with specified variance and adds to data
        noise = torch.randn_like(original_audio, device=device) * std_dev
        noisy_data = original_audio + noise

        # Specified that the data here needs to be dynamically clipped.
        domain_min = torch.min(original_audio)
        domain_max = torch.max(original_audio)
        
        # 3. Clip elementwise to X (as requested by Equation 9)
        noisy_data = torch.clamp(noisy_data, min=domain_min, max=domain_max)

    return noisy_data


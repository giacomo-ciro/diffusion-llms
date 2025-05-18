#!/usr/bin/env python3
"""
Generate multilingual EOS probability distribution plots by language and trigger words.
Plot shows X-axis token position, Y-axis probability of EOS token.
Analyzes English, Italian, and German with/without "explain" trigger words.

Enhanced visualization features:
- Statistical overlays (mean, median, percentiles)
- Focus plots on first N tokens
- Multi-language comparisons 
- Can run in visualization-only mode using pre-computed data
- Improved visualization with better readability
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import sys
import os
from tqdm import tqdm
import re
import argparse
from scipy import stats
sys.path.append('.')

# Set matplotlib parameters for better visualization
plt.rcParams['figure.figsize'] = (28, 16)  # Even larger default figure size
plt.rcParams['font.size'] = 18  # Larger base font size
plt.rcParams['axes.titlesize'] = 22  # Larger title
plt.rcParams['axes.labelsize'] = 20  # Larger axis labels
plt.rcParams['xtick.labelsize'] = 16  # Larger tick labels
plt.rcParams['ytick.labelsize'] = 16  # Larger tick labels
plt.rcParams['legend.fontsize'] = 18  # Larger legend text
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'  # Tight bounding box for saved figures
plt.rcParams['savefig.pad_inches'] = 0.5  # Add padding around the figure

# Import conditionally to allow for visualization-only mode
try:
    from diffusion_llms.models.llada import LladaBackbone
except ImportError:
    print("LladaBackbone import failed. Running in visualization-only mode.")

# Constants
LANGUAGES = ["english", "italian", "german"]
TRIGGER_WORDS = {
    "english": "explain", 
    "italian": "spiega", 
    "german": "erkläre"
}
N_SAMPLES = 10
SEQ_LEN = 1024
MASK_ID = 126336
OUTPUT_DIR = "./outputs/multilingual_eos"
DEFAULT_INPUT_DIR = "./outputs/multilingual_eos"  # Default directory for input data
DEFAULT_FOCUS_TOKENS = 25  # Default number of tokens to focus on in zoomed views

# Color schemes for consistent visualization
LANGUAGE_COLORS = {
    "english": "#1f77b4",  # Blue
    "italian": "#2ca02c",  # Green
    "german": "#d62728"    # Red
}

PROMPT_TYPE_COLORS = {
    "regular": "#1f77b4",  # Blue
    "trigger": "#ff7f0e"   # Orange
}

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def prepare_prompts():
    """Prepare prompts for each language with and without trigger words."""
    prompts = {}
    
    # English prompts
    english_regular = [
        "Tell me about climate change",
        "What is the capital of France",
        "How does photosynthesis work",
        "Define artificial intelligence",
        "Summarize the plot of Hamlet",
        "Who was Albert Einstein",
        "Describe the water cycle",
        "What happened during the French Revolution",
        "List the planets in our solar system",
        "How does the immune system function"
    ]
    
    english_trigger = [
        "Explain climate change",
        "Explain why Paris is the capital of France",
        "Explain photosynthesis",
        "Explain artificial intelligence",
        "Explain the plot of Hamlet",
        "Explain Albert Einstein's contributions",
        "Explain the water cycle",
        "Explain the causes of the French Revolution",
        "Explain our solar system's planets",
        "Explain how the immune system works"
    ]
    
    # Italian prompts
    italian_regular = [
        "Parlami del cambiamento climatico",
        "Qual è la capitale della Francia",
        "Come funziona la fotosintesi",
        "Definisci l'intelligenza artificiale",
        "Riassumi la trama di Amleto",
        "Chi era Albert Einstein",
        "Descrivi il ciclo dell'acqua",
        "Cosa è successo durante la Rivoluzione Francese",
        "Elenca i pianeti del nostro sistema solare",
        "Come funziona il sistema immunitario"
    ]
    
    italian_trigger = [
        "Spiega il cambiamento climatico",
        "Spiega perché Parigi è la capitale della Francia",
        "Spiega la fotosintesi",
        "Spiega l'intelligenza artificiale",
        "Spiega la trama di Amleto",
        "Spiega i contributi di Albert Einstein",
        "Spiega il ciclo dell'acqua",
        "Spiega le cause della Rivoluzione Francese",
        "Spiega i pianeti del nostro sistema solare",
        "Spiega come funziona il sistema immunitario"
    ]
    
    # German prompts
    german_regular = [
        "Erzähl mir über den Klimawandel",
        "Was ist die Hauptstadt von Frankreich",
        "Wie funktioniert die Photosynthese",
        "Definiere künstliche Intelligenz",
        "Fasse die Handlung von Hamlet zusammen",
        "Wer war Albert Einstein",
        "Beschreibe den Wasserkreislauf",
        "Was geschah während der Französischen Revolution",
        "Liste die Planeten in unserem Sonnensystem auf",
        "Wie funktioniert das Immunsystem"
    ]
    
    german_trigger = [
        "Erkläre den Klimawandel",
        "Erkläre warum Paris die Hauptstadt von Frankreich ist",
        "Erkläre die Photosynthese",
        "Erkläre künstliche Intelligenz",
        "Erkläre die Handlung von Hamlet",
        "Erkläre Albert Einsteins Beiträge",
        "Erkläre den Wasserkreislauf",
        "Erkläre die Ursachen der Französischen Revolution",
        "Erkläre die Planeten unseres Sonnensystems",
        "Erkläre wie das Immunsystem funktioniert"
    ]
    
    # Organize prompts by language and type
    prompts["english"] = {
        "regular": english_regular,
        "trigger": english_trigger
    }
    
    prompts["italian"] = {
        "regular": italian_regular,
        "trigger": italian_trigger
    }
    
    prompts["german"] = {
        "regular": german_regular,
        "trigger": german_trigger
    }
    
    return prompts

def get_eos_probabilities(model, prompt, seq_len=SEQ_LEN, mask_id=MASK_ID):
    """Process a prompt and return EOS token probabilities."""
    # Tokenize the prompt
    prompt_ids = model.tokenizer(prompt, return_tensors="pt").input_ids
    prompt_len = prompt_ids.shape[1]
    
    # Create input tensor with mask tokens
    x = torch.full((1, seq_len), mask_id, dtype=torch.long).to(model.device)
    
    # Replace the beginning of x with prompt tokens
    x[:, :prompt_len] = prompt_ids.clone()
    
    # Create a dummy target tensor
    dummy_target = torch.zeros_like(x)
    
    # Pass inputs to model and get logits
    model_output = model(x, target=dummy_target)
    logits = model_output['logits']
    
    # Extract EOS token logits and apply softmax
    eos_logits = logits[:, :, model.tokenizer.eos_token_id].squeeze(0)
    eos_probs = torch.nn.functional.softmax(eos_logits.to(torch.float64), dim=-1)
    
    # Zero out probs for already unmasked tokens
    eos_probs[:prompt_len] = 0
    
    # Detach and convert to numpy for analysis
    return eos_probs.detach().cpu().numpy()

def compute_statistics(probs_array):
    """Compute statistical measures for an array of probability distributions."""
    stats_dict = {
        "mean": np.mean(probs_array, axis=0),
        "median": np.median(probs_array, axis=0),
        "std": np.std(probs_array, axis=0),
        "p25": np.percentile(probs_array, 25, axis=0),
        "p75": np.percentile(probs_array, 75, axis=0),
        "max_pos": np.argmax(np.mean(probs_array, axis=0)),
        "max_val": np.max(np.mean(probs_array, axis=0))
    }
    return stats_dict

def add_statistics_to_plot(ax, stats_dict, color, alpha_fill=0.2, show_percentiles=False):
    """Add statistical information to an existing plot."""
    # Add mean +/- std dev range with improved visibility
    ax.fill_between(
        range(len(stats_dict["mean"])),
        stats_dict["mean"] - stats_dict["std"],
        stats_dict["mean"] + stats_dict["std"],
        alpha=alpha_fill,
        color=color,
        label="±1 std dev"
    )
    
    # Add 25-75 percentile range only if explicitly requested
    if show_percentiles:
        ax.fill_between(
            range(len(stats_dict["mean"])),
            stats_dict["p25"],
            stats_dict["p75"],
            alpha=alpha_fill*2.5,  # Increased alpha for better visibility
            color=color,
            label="25-75 percentile"
        )
    
    # Add enhanced annotations for max position
    max_pos = stats_dict["max_pos"]
    max_val = stats_dict["max_val"]
    
    # Intelligently position the annotation based on position in the sequence
    # to avoid cutoff at edges
    if max_pos > len(stats_dict["mean"]) * 0.7:
        # If near right edge, place annotation to the left
        xytext_pos = (max_pos-80, max_val*1.1)
    else:
        # Otherwise place to the right
        xytext_pos = (max_pos+20, max_val*1.1)
    
    ax.annotate(
        f"Max: {max_val:.2e} @ pos {max_pos}",
        xy=(max_pos, max_val),
        xytext=xytext_pos,
        arrowprops=dict(
            arrowstyle="->", 
            color=color, 
            lw=2.5,  # Even thicker arrow
            connectionstyle="arc3,rad=0.2",  # Curved arrow for better visibility
            alpha=0.9  # More visible
        ),
        color=color,
        fontsize=16,  # Even larger font
        bbox=dict(
            boxstyle="round,pad=0.4", 
            fc="white", 
            alpha=0.9, 
            ec=color
        )  # Background box
    )

def plot_language_comparison(all_results, prompt_type, focus_first_n=None, output_dir=OUTPUT_DIR):
    """Plot EOS probability comparison across languages for a given prompt type.
    
    Args:
        all_results: Dictionary with results for all languages
        prompt_type: Type of prompt ('regular' or 'trigger')
        focus_first_n: If set, creates a second plot focusing on first N tokens
        output_dir: Directory to save output plots
    """
    # Set explicit parameters for this plot
    plt.figure(figsize=(30, 18))  # Even larger figure for language comparisons
    sns.set(style="whitegrid")
    
    languages = list(all_results.keys())
    statistics = {}
    ax = plt.gca()
    
    for lang in languages:
        probs_array = all_results[lang][prompt_type]
        stats_dict = compute_statistics(probs_array)
        statistics[lang] = stats_dict
        
        # Plot mean line
        color = LANGUAGE_COLORS.get(lang, f"C{languages.index(lang)}")
        ax.plot(
            stats_dict["mean"], 
            label=f"{lang.capitalize()} (mean)",
            color=color,
            linewidth=4  # Increased line width for better visibility
        )
        
        # Add statistical overlays
        add_statistics_to_plot(ax, stats_dict, color, show_percentiles=False)
        
        # Mark maximum position
        argmax_pos = stats_dict["max_pos"]
        max_prob = stats_dict["max_val"]
        ax.plot(argmax_pos, max_prob, 'o', markersize=12, color=color)  # Larger marker
        ax.axvline(x=argmax_pos, linestyle='--', alpha=0.5, color=color, linewidth=2)  # Thicker line
    
    ax.set_title(f"EOS Probability per Position ({prompt_type.capitalize()} Prompts) - Language Comparison", fontsize=22, pad=20)
    ax.set_xlabel("Position", fontsize=20, labelpad=15)
    ax.set_ylabel("Probability", fontsize=20, labelpad=15)
    ax.set_xticks(np.arange(0, SEQ_LEN, step=50))
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xlim(0, SEQ_LEN)
    
    # Set y-limit based on maximum probability
    max_probs = [stats["max_val"] for stats in statistics.values()]
    ax.set_ylim(0, 1.2 * max(max_probs))
    
    # Create an enhanced legend with better positioning
    plt.legend(loc='upper left', fontsize=18, framealpha=0.9, frameon=True,
               facecolor='white', edgecolor='gray', fancybox=True, shadow=True)
    
    # Use subplots_adjust instead of tight_layout to prevent warnings
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.95, hspace=0.4, wspace=0.3)
    plt.savefig(os.path.join(output_dir, f"{prompt_type}_language_comparison.png"), 
               bbox_inches='tight', pad_inches=0.5, dpi=300)
    
    # Create a focused plot if requested
    if focus_first_n is not None and focus_first_n > 0:
        plt.figure(figsize=(28, 16))  # Larger figure for focused view
        ax = plt.gca()
        
        for lang in languages:
            stats_dict = statistics[lang]
            color = LANGUAGE_COLORS.get(lang, f"C{languages.index(lang)}")
            
            # Plot only the first n tokens
            focused_mean = stats_dict["mean"][:focus_first_n]
            ax.plot(
                range(focus_first_n), 
                focused_mean,
                label=f"{lang.capitalize()} (mean)",
                color=color,
                linewidth=4  # Increased line width for better visibility
            )
            
            # Add focused statistics
            focused_stats = {
                "mean": stats_dict["mean"][:focus_first_n],
                "median": stats_dict["median"][:focus_first_n],
                "std": stats_dict["std"][:focus_first_n],
                "p25": stats_dict["p25"][:focus_first_n],
                "p75": stats_dict["p75"][:focus_first_n],
                "max_pos": np.argmax(stats_dict["mean"][:focus_first_n]),
                "max_val": np.max(stats_dict["mean"][:focus_first_n])
            }
            
            add_statistics_to_plot(ax, focused_stats, color, show_percentiles=False)
        
        ax.set_title(f"EOS Probability - First {focus_first_n} Tokens ({prompt_type.capitalize()} Prompts)", 
                    fontsize=22, pad=20)
        ax.set_xlabel("Position", fontsize=20, labelpad=15)
        ax.set_ylabel("Probability", fontsize=20, labelpad=15)
        ax.set_xticks(np.arange(0, focus_first_n, step=5))
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_xlim(0, focus_first_n)
        
        # Adjust y-limits for better visualization
        max_focused_probs = [np.max(stats["mean"][:focus_first_n]) for stats in statistics.values()]
        ax.set_ylim(0, 1.2 * max(max_focused_probs))
        
        # Create an enhanced legend
        plt.legend(loc='upper left', fontsize=18, framealpha=0.9, frameon=True,
                  facecolor='white', edgecolor='gray', fancybox=True, shadow=True)
        
        # Use subplots_adjust instead of tight_layout to prevent warnings
        plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.95, hspace=0.4, wspace=0.3)
        plt.savefig(os.path.join(output_dir, f"{prompt_type}_language_comparison_first_{focus_first_n}.png"),
                   bbox_inches='tight', pad_inches=0.5, dpi=300)
    
    plt.close('all')
    
def plot_trigger_comparison(all_results, language, focus_first_n=None, output_dir=OUTPUT_DIR):
    """Plot EOS probability comparison between regular and trigger prompts for a language.
    
    Args:
        all_results: Dictionary with results for all languages
        language: Language to plot
        focus_first_n: If set, creates a second plot focusing on first N tokens
        output_dir: Directory to save output plots
    """
    # Reset matplotlib params to ensure consistency
    plt.rcParams.update({
        'figure.figsize': (32, 18),  # Extra large figure for language-specific comparisons
        'font.size': 18,  # Much larger base font size
        'axes.titlesize': 22, 
        'axes.labelsize': 20,
        'xtick.labelsize': 18, 
        'ytick.labelsize': 18,
        'legend.fontsize': 18,
    })
    
    plt.figure(figsize=(32, 18))  # Extra large figure for language-specific comparisons
    sns.set(style="whitegrid")
    ax = plt.gca()
    
    # Calculate statistics for both prompt types
    reg_stats = compute_statistics(all_results[language]["regular"])
    trig_stats = compute_statistics(all_results[language]["trigger"])
    
    # Get positions and values
    reg_argmax = reg_stats["max_pos"]
    reg_max = reg_stats["max_val"]
    trig_argmax = trig_stats["max_pos"]
    trig_max = trig_stats["max_val"]
    
    # Calculate position shift and probability difference
    pos_shift = trig_argmax - reg_argmax
    prob_diff = trig_max - reg_max
    reg_mean_prob = np.mean(reg_stats['mean'])
    trig_mean_prob = np.mean(trig_stats['mean'])
    
    # Plot mean lines with increased width for better visibility
    ax.plot(reg_stats["mean"], color=PROMPT_TYPE_COLORS["regular"], linewidth=4,
            label='Regular Prompts (mean)')
    ax.plot(trig_stats["mean"], color=PROMPT_TYPE_COLORS["trigger"], linewidth=4,
            label=f'"{TRIGGER_WORDS[language].capitalize()}" Prompts (mean)')
    
    # Add statistical overlays without percentile ranges
    add_statistics_to_plot(ax, reg_stats, PROMPT_TYPE_COLORS["regular"], show_percentiles=False)
    add_statistics_to_plot(ax, trig_stats, PROMPT_TYPE_COLORS["trigger"], show_percentiles=False)
    
    # Mark maximums with larger markers
    ax.plot(reg_argmax, reg_max, 'o', markersize=14, color=PROMPT_TYPE_COLORS["regular"])
    ax.plot(trig_argmax, trig_max, 'o', markersize=14, color=PROMPT_TYPE_COLORS["trigger"])
    
    # Add vertical lines with better visibility
    ax.axvline(x=reg_argmax, color=PROMPT_TYPE_COLORS["regular"], linestyle='--', linewidth=3, alpha=0.7,
              label=f'Regular Max: pos {reg_argmax}, prob {reg_max:.2e}')
    ax.axvline(x=trig_argmax, color=PROMPT_TYPE_COLORS["trigger"], linestyle='--', linewidth=3, alpha=0.7,
              label=f'Trigger Max: pos {trig_argmax}, prob {trig_max:.2e}')
    
    # Add a shaded region to highlight the shift
    min_y, max_y = ax.get_ylim()
    rect_height = max_y * 0.05  # 5% of max height
    rect_y = max_y * 0.9  # Position at 90% of max height
    
    # Draw arrow between the max positions
    if pos_shift != 0:
        arrow_props = dict(
            arrowstyle='<->', 
            lw=2, 
            color='black',
            shrinkA=5,
            shrinkB=5
        )
        ax.annotate('', 
                   xy=(reg_argmax, rect_y + rect_height/2),
                   xytext=(trig_argmax, rect_y + rect_height/2),
                   arrowprops=arrow_props)
        
        # Add label for the position shift
        shift_label_pos = (reg_argmax + trig_argmax) / 2
        ax.text(shift_label_pos, rect_y + rect_height*2,
               f"Position Shift: {pos_shift} tokens",
               ha='center', va='bottom', fontsize=14,
               bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9))
    
    # Add enhanced statistical measures to plot
    stats_info = (
        f"Position Shift: {pos_shift} tokens\n"
        f"Probability Diff: {prob_diff:.2e}\n"
        f"Regular Max: pos {reg_argmax}, prob {reg_max:.2e}\n"
        f"Trigger Max: pos {trig_argmax}, prob {trig_max:.2e}\n"
        f"Regular Mean: {reg_mean_prob:.2e}\n"
        f"Trigger Mean: {trig_mean_prob:.2e}"
    )
    
    # Position stats box in a better location based on language
    # For German and Italian which had cutoff issues, position differently
    if language in ["german", "italian"]:
        # Position in top right for languages that had cutoff issues
        stats_box_x, stats_box_y = 0.7, 0.98
    else:
        # Position in top left for other languages
        stats_box_x, stats_box_y = 0.02, 0.98
        
    ax.text(stats_box_x, stats_box_y, stats_info,
           transform=ax.transAxes, 
           bbox=dict(facecolor='white', alpha=0.9, boxstyle="round,pad=0.5", 
                   edgecolor="gray", linewidth=2),  # Thicker border
           verticalalignment='top',
           fontsize=18)
    
    # Create a title with the trigger word highlighted
    title = f"EOS Probability Comparison - {language.capitalize()} - Trigger Word: \"{TRIGGER_WORDS[language].capitalize()}\""
    ax.set_title(title, fontsize=22, pad=20)  # More padding to avoid overlap
    ax.set_xlabel("Position", fontsize=20, labelpad=15)
    ax.set_ylabel("Probability", fontsize=20, labelpad=15)
    ax.set_xticks(np.arange(0, SEQ_LEN, step=50))
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xlim(0, SEQ_LEN)
    ax.set_ylim(0, 1.2 * max(reg_max, trig_max))
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Create an enhanced legend with better positioning
    # For German and Italian, place the legend in a different location
    if language in ["german", "italian"]:
        legend_loc = 'upper center'  # Move legend to center to avoid overlap with stats box
    else:
        legend_loc = 'upper left'
        
    plt.legend(loc=legend_loc, fontsize=18, framealpha=0.9, frameon=True, 
               facecolor='white', edgecolor='gray', fancybox=True, shadow=True)
    
    # Use subplots_adjust instead of tight_layout to prevent warnings
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.95, hspace=0.4, wspace=0.3)
    plt.savefig(os.path.join(output_dir, f"{language}_trigger_comparison.png"), 
               bbox_inches='tight', pad_inches=0.5, dpi=300)  # Ensure nothing gets cut off
    
    # Create focused plot if requested
    if focus_first_n is not None and focus_first_n > 0:
        plt.figure(figsize=(24, 14))
        ax = plt.gca()
        
        # Get focused ranges
        reg_mean_focus = reg_stats["mean"][:focus_first_n]
        trig_mean_focus = trig_stats["mean"][:focus_first_n]
        
        # Create focused statistics
        reg_stats_focus = {
            "mean": reg_stats["mean"][:focus_first_n],
            "std": reg_stats["std"][:focus_first_n],
            "p25": reg_stats["p25"][:focus_first_n],
            "p75": reg_stats["p75"][:focus_first_n],
            "max_pos": np.argmax(reg_mean_focus),
            "max_val": np.max(reg_mean_focus)
        }
        
        trig_stats_focus = {
            "mean": trig_stats["mean"][:focus_first_n],
            "std": trig_stats["std"][:focus_first_n],
            "p25": trig_stats["p25"][:focus_first_n],
            "p75": trig_stats["p75"][:focus_first_n],
            "max_pos": np.argmax(trig_mean_focus),
            "max_val": np.max(trig_mean_focus)
        }
        
        # Calculate focused position shift
        focus_pos_shift = trig_stats_focus["max_pos"] - reg_stats_focus["max_pos"]
        focus_prob_diff = trig_stats_focus["max_val"] - reg_stats_focus["max_val"]
        
        # Plot focused means
        ax.plot(range(focus_first_n), reg_mean_focus, 
               color=PROMPT_TYPE_COLORS["regular"], linewidth=3,
               label='Regular Prompts (mean)')
        ax.plot(range(focus_first_n), trig_mean_focus, 
               color=PROMPT_TYPE_COLORS["trigger"], linewidth=3,
               label=f'"{TRIGGER_WORDS[language].capitalize()}" Prompts (mean)')
        
        # Add statistical overlays for the focused view
        add_statistics_to_plot(ax, reg_stats_focus, PROMPT_TYPE_COLORS["regular"], show_percentiles=False)
        add_statistics_to_plot(ax, trig_stats_focus, PROMPT_TYPE_COLORS["trigger"], show_percentiles=False)
        
        # Mark focused maximums
        ax.plot(reg_stats_focus["max_pos"], reg_stats_focus["max_val"], 
               'o', markersize=10, color=PROMPT_TYPE_COLORS["regular"])
        ax.plot(trig_stats_focus["max_pos"], trig_stats_focus["max_val"], 
               'o', markersize=10, color=PROMPT_TYPE_COLORS["trigger"])
        
        # Add vertical lines for focused view
        ax.axvline(x=reg_stats_focus["max_pos"], color=PROMPT_TYPE_COLORS["regular"], 
                  linestyle='--', linewidth=2, alpha=0.7)
        ax.axvline(x=trig_stats_focus["max_pos"], color=PROMPT_TYPE_COLORS["trigger"], 
                  linestyle='--', linewidth=2, alpha=0.7)
        
        # Add statistical measures for focused view
        focused_stats_info = (
            f"Position Shift: {focus_pos_shift} tokens\n"
            f"Probability Diff: {focus_prob_diff:.2e}\n"
            f"Regular Max: pos {reg_stats_focus['max_pos']}, prob {reg_stats_focus['max_val']:.2e}\n"
            f"Trigger Max: pos {trig_stats_focus['max_pos']}, prob {trig_stats_focus['max_val']:.2e}\n"
            f"Regular Mean: {np.mean(reg_mean_focus):.2e}\n"
            f"Trigger Mean: {np.mean(trig_mean_focus):.2e}"
        )
        
        # Position stats box in a better location based on language
        # For German and Italian which had cutoff issues, position differently
        if language in ["german", "italian"]:
            # Position in top right for languages that had cutoff issues
            stats_box_x, stats_box_y = 0.7, 0.98
        else:
            # Position in top left for other languages
            stats_box_x, stats_box_y = 0.02, 0.98
            
        ax.text(stats_box_x, stats_box_y, focused_stats_info,
               transform=ax.transAxes, 
               bbox=dict(facecolor='white', alpha=0.9, boxstyle="round,pad=0.5", 
                       edgecolor="gray", linewidth=2),  # Thicker border
               verticalalignment='top',
               fontsize=16)  # Larger font size
        
        # Set labels and title for focused view
        title = f"EOS Probability - First {focus_first_n} Tokens - {language.capitalize()} - Trigger: \"{TRIGGER_WORDS[language].capitalize()}\""
        ax.set_title(title, fontsize=20, pad=20)  # More padding to avoid overlap
        ax.set_xlabel("Position", fontsize=18, labelpad=15)
        ax.set_ylabel("Probability", fontsize=18, labelpad=15)
        ax.set_xticks(np.arange(0, focus_first_n, step=5))
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlim(0, focus_first_n)
        
        # Add grid for better readability in focused view
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set y-limit for focused view
        max_focused = max(np.max(reg_mean_focus), np.max(trig_mean_focus))
        ax.set_ylim(0, 1.2 * max_focused)
        
        # Create an enhanced legend for focused view with better positioning
        # For German and Italian, place the legend in a different location
        if language in ["german", "italian"]:
            legend_loc = 'upper center'  # Move legend to center to avoid overlap with stats box
        else:
            legend_loc = 'upper left'
            
        plt.legend(loc=legend_loc, fontsize=16, framealpha=0.9, frameon=True, 
                  facecolor='white', edgecolor='gray', fancybox=True, shadow=True)
        
        # Use subplots_adjust instead of tight_layout to prevent warnings
        plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.95, hspace=0.4, wspace=0.3)
        plt.savefig(os.path.join(output_dir, f"{language}_trigger_comparison_first_{focus_first_n}.png"), 
                   bbox_inches='tight', pad_inches=0.5, dpi=300)
    
    plt.close('all')
    
    return trig_argmax - reg_argmax  # Return the position shift

def plot_pointwise_difference(all_results, focus_first_n=None, output_dir=OUTPUT_DIR):
    """Plot pointwise difference between trigger and regular prompts across languages.
    
    Args:
        all_results: Dictionary with results for all languages
        focus_first_n: If set, creates a second plot focusing on first N tokens
        output_dir: Directory to save output plots
    """
    plt.figure(figsize=(28, 16))  # Larger figure for pointwise difference plot
    sns.set(style="whitegrid")
    ax = plt.gca()
    
    languages = list(all_results.keys())
    diffs = {}
    stats = {}
    
    for lang in languages:
        reg_avg = np.mean(all_results[lang]["regular"], axis=0)
        trig_avg = np.mean(all_results[lang]["trigger"], axis=0)
        diff = trig_avg - reg_avg
        diffs[lang] = diff
        
        # Calculate statistics for the difference
        mean_diff = np.mean(diff)
        median_diff = np.median(diff)
        max_diff = np.max(diff)
        max_diff_pos = np.argmax(diff)
        min_diff = np.min(diff)
        min_diff_pos = np.argmin(diff)
        
        stats[lang] = {
            "mean_diff": mean_diff,
            "median_diff": median_diff,
            "max_diff": max_diff,
            "max_diff_pos": max_diff_pos,
            "min_diff": min_diff,
            "min_diff_pos": min_diff_pos
        }
        
        # Plot the difference with language-specific color
        color = LANGUAGE_COLORS.get(lang, f"C{languages.index(lang)}")
        ax.plot(diff, label=f"{lang.capitalize()}", color=color, linewidth=4)
        
        # Add annotations for max and min differences
        ax.annotate(
            f"Max: {max_diff:.2e}",
            xy=(max_diff_pos, max_diff),
            xytext=(max_diff_pos+20, max_diff*1.1),
            arrowprops=dict(arrowstyle="->", color=color),
            color=color
        )
        
        ax.annotate(
            f"Min: {min_diff:.2e}",
            xy=(min_diff_pos, min_diff),
            xytext=(min_diff_pos+20, min_diff*1.1),
            arrowprops=dict(arrowstyle="->", color=color),
            color=color
        )
    
    # Add a line at y=0
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add statistical summary
    stat_text = "Statistical Summary:\n" + "\n".join(
        f"{lang.capitalize()}: Mean diff = {stats[lang]['mean_diff']:.2e}, "
        f"Median diff = {stats[lang]['median_diff']:.2e}"
        for lang in languages
    )
    
    ax.text(0.02, 0.98, stat_text,
           transform=ax.transAxes, 
           bbox=dict(facecolor='white', alpha=0.9, boxstyle="round,pad=0.5", 
                    edgecolor="gray", linewidth=2),  # Improved text box
           verticalalignment='top',
           fontsize=16)
    
    ax.set_title("Pointwise Difference in EOS Probabilities (Trigger - Regular)", fontsize=22, pad=20)
    ax.set_xlabel("Position", fontsize=20, labelpad=15)
    ax.set_ylabel("Probability Difference", fontsize=20, labelpad=15)
    ax.set_xticks(np.arange(0, SEQ_LEN, step=50))
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xlim(0, SEQ_LEN)
    
    # Set y-limits to make the zero line centered
    y_max = max(abs(np.max([np.max(diff) for diff in diffs.values()])), 
               abs(np.min([np.min(diff) for diff in diffs.values()])))
    ax.set_ylim(-1.2 * y_max, 1.2 * y_max)
    
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pointwise_difference.png"))
    
    # Create focused plot if requested
    if focus_first_n is not None and focus_first_n > 0:
        plt.figure(figsize=(20, 10))
        ax = plt.gca()
        
        for lang in languages:
            # Get focused difference
            focused_diff = diffs[lang][:focus_first_n]
            color = LANGUAGE_COLORS.get(lang, f"C{languages.index(lang)}")
            
            # Plot focused difference
            ax.plot(range(focus_first_n), focused_diff, 
                  label=f"{lang.capitalize()}", color=color, linewidth=4)
            
            # Calculate statistics for the focused region
            max_diff_focus = np.max(focused_diff)
            max_diff_pos_focus = np.argmax(focused_diff)
            min_diff_focus = np.min(focused_diff)
            min_diff_pos_focus = np.argmin(focused_diff)
            
            # Add annotations
            ax.annotate(
                f"Max: {max_diff_focus:.2e}",
                xy=(max_diff_pos_focus, max_diff_focus),
                xytext=(max_diff_pos_focus+2, max_diff_focus*1.1),
                arrowprops=dict(arrowstyle="->", color=color),
                color=color
            )
            
            ax.annotate(
                f"Min: {min_diff_focus:.2e}",
                xy=(min_diff_pos_focus, min_diff_focus),
                xytext=(min_diff_pos_focus+2, min_diff_focus*1.1),
                arrowprops=dict(arrowstyle="->", color=color),
                color=color
            )
        
        # Add a line at y=0
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Set labels and title
        ax.set_title(f"Pointwise Difference - First {focus_first_n} Tokens")
        ax.set_xlabel("Position")
        ax.set_ylabel("Probability Difference")
        ax.set_xticks(np.arange(0, focus_first_n, step=5))
        ax.set_xlim(0, focus_first_n)
        
        # Set y-limits for focused view
        y_max_focus = max([max(abs(np.max(diffs[lang][:focus_first_n])), 
                             abs(np.min(diffs[lang][:focus_first_n]))) 
                          for lang in languages])
        ax.set_ylim(-1.2 * y_max_focus, 1.2 * y_max_focus)
        
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"pointwise_difference_first_{focus_first_n}.png"))
    
    plt.close('all')
    return diffs

def plot_combined_comparison(all_results, focus_first_n=None, output_dir=OUTPUT_DIR, show_avg_median=True):
    """Create a combined plot with all languages and prompt types.
    
    Args:
        all_results: Dictionary with results for all languages
        focus_first_n: If set, creates a second plot focusing on first N tokens
        output_dir: Directory to save output plots
        show_avg_median: If True, also show vertical lines for average and median positions
    """
    # Reset matplotlib params to ensure consistency 
    plt.rcParams.update({
        'figure.figsize': (32, 18),  # Extra large figure for combined comparison
        'font.size': 16,  # Larger font
        'axes.titlesize': 22, 
        'axes.labelsize': 20,
        'xtick.labelsize': 16, 
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
    })
    
    plt.figure(figsize=(32, 18))  # Extra large figure to fit all languages
    sns.set(style="whitegrid")
    ax = plt.gca()
    
    languages = list(all_results.keys())
    
    # Dictionary to store max values for vertical lines
    max_positions = {}
    avg_positions = {}
    median_positions = {}
    
    # Statistical summary data
    summary_data = []
    
    # Plot each language and prompt type combination
    for lang in languages:
        for prompt_type in ["regular", "trigger"]:
            stats = compute_statistics(all_results[lang][prompt_type])
            
            # Determine line style and color
            ls = '-' if prompt_type == "regular" else '--'
            color = LANGUAGE_COLORS.get(lang, f"C{languages.index(lang)}")
            
            # Plot the mean
            ax.plot(
                stats["mean"], 
                linestyle=ls,
                color=color,
                linewidth=3,  # Increased line width for better visibility
                label=f"{lang.capitalize()} ({prompt_type})"
            )
            
            # Mark maximum position
            max_pos = stats["max_pos"]
            max_val = stats["max_val"]
            max_positions[f"{lang}_{prompt_type}"] = {"pos": max_pos, "val": max_val}
            
            # Mark the max with a dot
            marker = 'o' if prompt_type == "regular" else 's'
            ax.plot(max_pos, max_val, marker=marker, markersize=10, color=color)  # Increased marker size
            
            # Calculate average and median positions
            if show_avg_median:
                # Weighted average position (weighted by probability)
                avg_pos = np.sum(np.arange(len(stats["mean"])) * stats["mean"]) / np.sum(stats["mean"])
                avg_positions[f"{lang}_{prompt_type}"] = {"pos": avg_pos, "val": stats["mean"][int(avg_pos)]}
                
                # Median position of max probabilities across samples
                median_pos = np.median(np.argmax(all_results[lang][prompt_type], axis=1))
                median_positions[f"{lang}_{prompt_type}"] = {"pos": median_pos, "val": stats["mean"][int(median_pos)]}
            
            # Collect summary data
            summary_data.append({
                "Language": lang.capitalize(),
                "Type": prompt_type.capitalize(),
                "Max Position": int(max_pos),
                "Max Value": f"{max_val:.6f}",
                "Avg Position": int(avg_pos) if show_avg_median else "N/A",
                "Median Position": int(median_pos) if show_avg_median else "N/A",
                "Mean Probability": f"{np.mean(stats['mean']):.6f}"
            })
    
    # Add vertical lines at max positions
    for key, data in max_positions.items():
        lang, prompt_type = key.split('_')
        ls = '-' if prompt_type == "regular" else '--'
        color = LANGUAGE_COLORS.get(lang, f"C{languages.index(lang)}")
        
        ax.axvline(x=data["pos"], color=color, linestyle=ls, alpha=0.4)
    
    # Add vertical lines for average positions if requested
    if show_avg_median:
        for key, data in avg_positions.items():
            lang, prompt_type = key.split('_')
            color = LANGUAGE_COLORS.get(lang, f"C{languages.index(lang)}")
            
            # Dotted line for average position
            ax.axvline(x=data["pos"], color=color, linestyle=':', alpha=0.7)
            
            # Add annotation for average position with intelligent positioning
            # Offset positions for Italian and German to avoid overlap with other annotations
            offset_x = 10
            offset_y_factor = 0.1
            
            # Handle special cases for German and Italian which had issues with cutoff
            if lang in ["german", "italian"]:
                # Adjust position based on language to avoid overlap
                if lang == "german":
                    offset_x = -30 if prompt_type == "regular" else 30
                    offset_y_factor = 0.15
                elif lang == "italian":
                    offset_x = -25 if prompt_type == "trigger" else 25
                    offset_y_factor = 0.12
            
            ax.annotate(
                f"{lang.capitalize()} {prompt_type} avg",
                xy=(data["pos"], data["val"]),
                xytext=(data["pos"] + offset_x, data["val"] + offset_y_factor * data["val"]),
                arrowprops=dict(
                    arrowstyle="->", 
                    color=color, 
                    alpha=0.7, 
                    lw=2,
                    connectionstyle="arc3,rad=0.2"  # Curved arrow for better visibility
                ),
                color=color,
                fontsize=14,  # Larger font
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec=color)  # Background box with border color
            )
        
        # Add vertical lines for median positions
        for key, data in median_positions.items():
            lang, prompt_type = key.split('_')
            color = LANGUAGE_COLORS.get(lang, f"C{languages.index(lang)}")
            
            # Dash-dotted line for median position
            ax.axvline(x=data["pos"], color=color, linestyle='-.', alpha=0.7)
            
            # Add annotation for median position with intelligent positioning
            # Offset positions for Italian and German to avoid overlap with other annotations
            offset_x = -10
            offset_y_factor = -0.2
            
            # Handle special cases for German and Italian which had issues with cutoff
            if lang in ["german", "italian"]:
                # Adjust position based on language to avoid overlap
                if lang == "german": 
                    offset_x = 35 if prompt_type == "regular" else -35
                    offset_y_factor = -0.25
                elif lang == "italian":
                    offset_x = 30 if prompt_type == "trigger" else -30
                    offset_y_factor = -0.22
            
            ax.annotate(
                f"{lang.capitalize()} {prompt_type} med",
                xy=(data["pos"], data["val"]),
                xytext=(data["pos"] + offset_x, data["val"] + offset_y_factor * data["val"]),
                arrowprops=dict(
                    arrowstyle="->", 
                    color=color, 
                    alpha=0.7, 
                    lw=2,
                    connectionstyle="arc3,rad=-0.2"  # Curved arrow in opposite direction from avg
                ),
                color=color,
                fontsize=14,  # Larger font
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec=color)  # Background box with border color
            )
    
    # Add statistical summary table with better styling
    table_text = "Statistical Summary:\n"
    for item in summary_data:
        table_text += f"{item['Language']} ({item['Type']}): Max @ pos {item['Max Position']} ({item['Max Value']}), "
        if show_avg_median:
            table_text += f"Avg pos: {item['Avg Position']}, Med pos: {item['Median Position']}, "
        table_text += f"Mean prob: {item['Mean Probability']}\n"
    
    # Place the summary in a text box above the plot with improved styling
    props = dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='gray', linewidth=2)
    ax.text(0.5, 1.05, table_text, transform=ax.transAxes, fontsize=16,
            verticalalignment='bottom', horizontalalignment='center',
            bbox=props)
    
    ax.set_title(f"Combined EOS Probability Comparison - All Languages & Prompt Types", 
                fontsize=22, pad=100)  # Extra padding for title to make room for summary table
    ax.set_xlabel("Position", fontsize=20, labelpad=15)
    ax.set_ylabel("Probability", fontsize=20, labelpad=15)
    ax.set_xticks(np.arange(0, SEQ_LEN, step=50))
    ax.tick_params(axis='both', which='major', labelsize=16)  # Larger tick labels
    ax.set_xlim(0, SEQ_LEN)
    
    # Find overall maximum for y-limit
    all_max_vals = [data["val"] for data in max_positions.values()]
    ax.set_ylim(0, 1.2 * max(all_max_vals))
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Create a larger and more prominent legend with better positioning
    plt.legend(loc='upper left', fontsize=16, framealpha=0.9, frameon=True, 
               facecolor='white', edgecolor='gray', fancybox=True, shadow=True)
    
    plt.tight_layout(pad=3.0)  # Add extra padding
    plt.subplots_adjust(top=0.85, bottom=0.1, left=0.1, right=0.95)  # Make room for the summary table and adjust margins
    plt.savefig(os.path.join(output_dir, "combined_comparison.png"), 
               bbox_inches='tight', pad_inches=0.5, dpi=300)  # Ensure nothing gets cut off
    
    # Create focused plot if requested
    if focus_first_n is not None and focus_first_n > 0:
        plt.figure(figsize=(24, 14))
        ax = plt.gca()
        
        # Dictionary to store focused positions
        focused_max_positions = {}
        focused_avg_positions = {}
        focused_median_positions = {}
        focused_summary_data = []
        
        for lang in languages:
            for prompt_type in ["regular", "trigger"]:
                # Extract data for the first N tokens
                mean_vals = np.mean(all_results[lang][prompt_type], axis=0)[:focus_first_n]
                
                # Determine line style and color
                ls = '-' if prompt_type == "regular" else '--'
                color = LANGUAGE_COLORS.get(lang, f"C{languages.index(lang)}")
                
                # Plot focused data
                ax.plot(
                    range(focus_first_n),
                    mean_vals, 
                    linestyle=ls,
                    color=color,
                    linewidth=3,  # Thicker line
                    label=f"{lang.capitalize()} ({prompt_type})"
                )
                
                # Mark local maximum in focused region
                max_pos_focus = np.argmax(mean_vals)
                max_val_focus = np.max(mean_vals)
                focused_max_positions[f"{lang}_{prompt_type}"] = {"pos": max_pos_focus, "val": max_val_focus}
                
                marker = 'o' if prompt_type == "regular" else 's'
                ax.plot(max_pos_focus, max_val_focus, marker=marker, markersize=10, color=color)  # Larger marker
                
                # Calculate average and median positions within the focused region
                if show_avg_median:
                    # Weighted average position (weighted by probability)
                    if np.sum(mean_vals) > 0:  # Avoid division by zero
                        avg_pos_focus = np.sum(np.arange(len(mean_vals)) * mean_vals) / np.sum(mean_vals)
                        focused_avg_positions[f"{lang}_{prompt_type}"] = {
                            "pos": avg_pos_focus, 
                            "val": mean_vals[int(avg_pos_focus)]
                        }
                    
                    # Median position of max probabilities across samples within focus area
                    focused_samples = all_results[lang][prompt_type][:, :focus_first_n]
                    max_positions_focused = np.argmax(focused_samples, axis=1)
                    median_pos_focus = np.median(max_positions_focused)
                    focused_median_positions[f"{lang}_{prompt_type}"] = {
                        "pos": median_pos_focus,
                        "val": mean_vals[int(median_pos_focus)]
                    }
                
                # Collect focused summary data
                focused_summary_data.append({
                    "Language": lang.capitalize(),
                    "Type": prompt_type.capitalize(),
                    "Max Position": int(max_pos_focus),
                    "Max Value": f"{max_val_focus:.6f}",
                    "Avg Position": int(avg_pos_focus) if show_avg_median and 'avg_pos_focus' in locals() else "N/A",
                    "Median Position": int(median_pos_focus) if show_avg_median else "N/A",
                    "Mean Probability": f"{np.mean(mean_vals):.6f}"
                })
        
        # Add vertical lines for maximum positions in focused view
        for key, data in focused_max_positions.items():
            lang, prompt_type = key.split('_')
            ls = '-' if prompt_type == "regular" else '--'
            color = LANGUAGE_COLORS.get(lang, f"C{languages.index(lang)}")
            
            ax.axvline(x=data["pos"], color=color, linestyle=ls, alpha=0.4)
        
        # Add vertical lines and annotations for average and median positions if requested
        if show_avg_median:
            for key, data in focused_avg_positions.items():
                lang, prompt_type = key.split('_')
                color = LANGUAGE_COLORS.get(lang, f"C{languages.index(lang)}")
                
                # Dotted line for average position
                ax.axvline(x=data["pos"], color=color, linestyle=':', alpha=0.7)
                
                # Add annotation for average position with intelligent positioning
                # For focused view, use smaller offsets but similar logic
                offset_x = 2
                offset_y_factor = 0.1
                
                # Handle special cases for German and Italian which had issues with cutoff
                if lang in ["german", "italian"]:
                    if lang == "german":
                        offset_x = -4 if prompt_type == "regular" else 4
                        offset_y_factor = 0.15
                    elif lang == "italian":
                        offset_x = -3 if prompt_type == "trigger" else 3
                        offset_y_factor = 0.12
                        
                ax.annotate(
                    f"{lang.capitalize()} {prompt_type} avg",
                    xy=(data["pos"], data["val"]),
                    xytext=(data["pos"] + offset_x, data["val"] + offset_y_factor * data["val"]),
                    arrowprops=dict(
                        arrowstyle="->", 
                        color=color, 
                        alpha=0.7, 
                        lw=2,
                        connectionstyle="arc3,rad=0.2"
                    ),
                    color=color,
                    fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec=color)
                )
            
            for key, data in focused_median_positions.items():
                lang, prompt_type = key.split('_')
                color = LANGUAGE_COLORS.get(lang, f"C{languages.index(lang)}")
                
                # Dash-dotted line for median position
                ax.axvline(x=data["pos"], color=color, linestyle='-.', alpha=0.7)
                
                # Add annotation for median position with intelligent positioning
                # For focused view, use smaller offsets
                offset_x = -2
                offset_y_factor = -0.2
                
                # Handle special cases for German and Italian which had issues with cutoff
                if lang in ["german", "italian"]:
                    if lang == "german":
                        offset_x = 4 if prompt_type == "regular" else -4
                        offset_y_factor = -0.25
                    elif lang == "italian":
                        offset_x = 3 if prompt_type == "trigger" else -3
                        offset_y_factor = -0.22
                
                ax.annotate(
                    f"{lang.capitalize()} {prompt_type} med",
                    xy=(data["pos"], data["val"]),
                    xytext=(data["pos"] + offset_x, data["val"] + offset_y_factor * data["val"]),
                    arrowprops=dict(
                        arrowstyle="->", 
                        color=color, 
                        alpha=0.7, 
                        lw=2,
                        connectionstyle="arc3,rad=-0.2"
                    ),
                    color=color,
                    fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec=color)
                )
        
        # Add statistical summary table for focused view
        focused_table_text = f"Statistical Summary (First {focus_first_n} Tokens):\n"
        for item in focused_summary_data:
            focused_table_text += f"{item['Language']} ({item['Type']}): Max @ pos {item['Max Position']} ({item['Max Value']}), "
            if show_avg_median:
                focused_table_text += f"Avg pos: {item['Avg Position']}, Med pos: {item['Median Position']}, "
            focused_table_text += f"Mean prob: {item['Mean Probability']}\n"
        
        # Place the summary in a text box with improved styling
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=2)
        ax.text(0.5, 1.04, focused_table_text, transform=ax.transAxes, fontsize=14,
                verticalalignment='bottom', horizontalalignment='center',
                bbox=props)
        
        ax.set_title(f"Combined EOS Probability - First {focus_first_n} Tokens", fontsize=22, pad=80)
        ax.set_xlabel("Position", fontsize=20, labelpad=15)
        ax.set_ylabel("Probability", fontsize=20, labelpad=15)
        ax.set_xticks(np.arange(0, focus_first_n, step=5))
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_xlim(0, focus_first_n)
        
        # Add grid for better readability
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Find max value in the focused region for all plots
        all_max = max([np.max(np.mean(all_results[lang][pt], axis=0)[:focus_first_n]) 
                     for lang in languages for pt in ["regular", "trigger"]])
        ax.set_ylim(0, 1.2 * all_max)
        
        # Create a larger and more prominent legend with better positioning
        plt.legend(loc='upper left', fontsize=16, framealpha=0.9, frameon=True, 
                  facecolor='white', edgecolor='gray', fancybox=True, shadow=True)
        
        plt.tight_layout(pad=3.0)  # Extra padding to prevent cutoff
        plt.subplots_adjust(top=0.85, bottom=0.1, left=0.1, right=0.95)  # Make room for the summary table and adjust margins
        plt.savefig(os.path.join(output_dir, f"combined_comparison_first_{focus_first_n}.png"), 
                   bbox_inches='tight', pad_inches=0.5, dpi=300)  # Ensure nothing gets cut off
    
    plt.close('all')

def load_results_from_files(input_dir=DEFAULT_INPUT_DIR):
    """Load pre-computed results from NPY files."""
    print(f"Loading pre-computed results from {input_dir}")
    results = {}
    
    for language in LANGUAGES:
        results[language] = {}
        
        for prompt_type in ["regular", "trigger"]:
            file_path = os.path.join(input_dir, f"{language}_{prompt_type}.npy")
            
            if os.path.exists(file_path):
                results[language][prompt_type] = np.load(file_path)
                print(f"Loaded {file_path}")
            else:
                print(f"Warning: File not found: {file_path}")
                results[language][prompt_type] = []
    
    return results

def create_visualizations(results, focus_first_n=None, output_dir=OUTPUT_DIR, show_avg_median=True):
    """Generate all visualizations from pre-computed results.
    
    Args:
        results: Dictionary with results for all languages
        focus_first_n: If set, creates a second plot focusing on first N tokens
        output_dir: Directory to save output plots
        show_avg_median: If True, also show vertical lines for average and median positions
    """
    print("\nGenerating visualizations...")
    if show_avg_median:
        print("Including average and median position indicators")
    
    # Compare languages for each prompt type
    for prompt_type in ["regular", "trigger"]:
        plot_language_comparison(results, prompt_type, focus_first_n, output_dir)
    
    # Compare regular vs trigger for each language
    position_shifts = {}
    for language in LANGUAGES:
        if results[language]["regular"].size > 0 and results[language]["trigger"].size > 0:
            position_shifts[language] = plot_trigger_comparison(
                results, language, focus_first_n, output_dir)
    
    # Analyze pointwise differences
    diffs = plot_pointwise_difference(results, focus_first_n, output_dir)
    
    # Create combined comparison
    plot_combined_comparison(results, focus_first_n, output_dir, show_avg_median)
    
    # Print summary statistics
    print("\nSUMMARY OF RESULTS:")
    print("===================")
    print("EOS Position Shifts (Trigger - Regular):")
    for lang, shift in position_shifts.items():
        print(f"  {lang.capitalize()}: {shift} positions")
    
    print("\nAverage Max EOS Probabilities:")
    for lang in LANGUAGES:
        if results[lang]["regular"].size > 0:
            reg_avg = np.mean(results[lang]["regular"], axis=0)
            reg_max = np.max(reg_avg)
            reg_argmax = np.argmax(reg_avg)
            print(f"  {lang.capitalize()}, Regular: {reg_max:.4e} at position {reg_argmax}")
            
        if results[lang]["trigger"].size > 0:
            trig_avg = np.mean(results[lang]["trigger"], axis=0)
            trig_max = np.max(trig_avg)
            trig_argmax = np.argmax(trig_avg)
            print(f"  {lang.capitalize()}, Trigger: {trig_max:.4e} at position {trig_argmax}")
    
    print(f"\nDone! All visualizations saved to: {output_dir}")
    return position_shifts, diffs

def main():
    """Main function to run the EOS probability analysis."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate EOS probability distribution plots by language and trigger words."
    )
    parser.add_argument(
        "--mode", type=str, choices=["full", "visualize"], default="full",
        help="Run mode: 'full' to generate data and visualize, 'visualize' to only create plots from existing data"
    )
    parser.add_argument(
        "--input-dir", type=str, default=DEFAULT_INPUT_DIR,
        help=f"Directory containing pre-computed .npy files (for 'visualize' mode)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=OUTPUT_DIR,
        help=f"Directory to save generated plots"
    )
    parser.add_argument(
        "--focus", type=int, default=DEFAULT_FOCUS_TOKENS,
        help=f"Number of tokens to focus on in zoomed views (default: {DEFAULT_FOCUS_TOKENS})"
    )
    parser.add_argument(
        "--show-avg-median", action="store_true", default=True,
        help="Show vertical lines for average and median positions (default: True)"
    )
    parser.add_argument(
        "--no-avg-median", dest="show_avg_median", action="store_false",
        help="Don't show vertical lines for average and median positions"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == "full":
        print("Running in FULL mode: generating data and visualizations")
        print("Loading model...")
        device = 'mps'  # Adjust as needed for your environment
        
        # Clear CUDA cache if using GPU
        torch.mps.empty_cache()
        
        # Instantiate model
        model = LladaBackbone()
        
        # Prepare prompts
        print("Preparing prompts...")
        prompts = prepare_prompts()
        
        # Initialize results dictionary
        results = {}
        
        # Process all languages and prompt types
        print("Processing prompts and collecting EOS probabilities...")
        for language in LANGUAGES:
            print(f"\nProcessing {language.upper()}")
            results[language] = {
                "regular": [],
                "trigger": []
            }
            
            for prompt_type in ["regular", "trigger"]:
                print(f"  Processing {prompt_type} prompts...")
                
                for i, prompt in enumerate(tqdm(prompts[language][prompt_type])):
                    eos_probs = get_eos_probabilities(model, prompt)
                    results[language][prompt_type].append(eos_probs)
                    
                    # Display some stats
                    argmax_pos = np.argmax(eos_probs)
                    max_prob = eos_probs[argmax_pos]
                    print(f"    - Prompt {i+1}: '{prompt}'")
                    print(f"      Max EOS prob: {max_prob:.4e} at position {argmax_pos}")
                
                # Save the result array
                output_path = os.path.join(args.output_dir, f"{language}_{prompt_type}.npy")
                np.save(output_path, np.array(results[language][prompt_type]))
        
        # Generate visualizations
        create_visualizations(results, args.focus, args.output_dir, args.show_avg_median)
        
    else:  # Visualization-only mode
        print("Running in VISUALIZE mode: creating plots from existing data")
        # Load pre-computed results
        results = load_results_from_files(args.input_dir)
        
        # Generate visualizations
        create_visualizations(results, args.focus, args.output_dir, args.show_avg_median)

if __name__ == "__main__":
    main()

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

from diffusion_llms.models.llada import LladaBackbone

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

def add_statistics_to_plot(ax, stats_dict, color, alpha_fill=0.2):
    """Add statistical information to an existing plot."""
    # Add mean +/- std dev range
    ax.fill_between(
        range(len(stats_dict["mean"])),
        stats_dict["mean"] - stats_dict["std"],
        stats_dict["mean"] + stats_dict["std"],
        alpha=alpha_fill,
        color=color,
        label="±1 std dev"
    )
    
    # Add 25-75 percentile range with different alpha
    ax.fill_between(
        range(len(stats_dict["mean"])),
        stats_dict["p25"],
        stats_dict["p75"],
        alpha=alpha_fill*2,
        color=color,
        label="25-75 percentile"
    )
    
    # Add annotations for max position
    max_pos = stats_dict["max_pos"]
    max_val = stats_dict["max_val"]
    
    ax.annotate(
        f"Max: {max_val:.2e} @ pos {max_pos}",
        xy=(max_pos, max_val),
        xytext=(max_pos+20, max_val*1.1),
        arrowprops=dict(arrowstyle="->", color=color),
        color=color
    )

def plot_language_comparison(all_results, prompt_type, focus_first_n=None, output_dir=OUTPUT_DIR):
    """Plot EOS probability comparison across languages for a given prompt type.
    
    Args:
        all_results: Dictionary with results for all languages
        prompt_type: Type of prompt ('regular' or 'trigger')
        focus_first_n: If set, creates a second plot focusing on first N tokens
        output_dir: Directory to save output plots
    """
    plt.figure(figsize=(20, 10))
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
            linewidth=2
        )
        
        # Add statistical overlays
        add_statistics_to_plot(ax, stats_dict, color)
        
        # Mark maximum position
        argmax_pos = stats_dict["max_pos"]
        max_prob = stats_dict["max_val"]
        ax.plot(argmax_pos, max_prob, 'o', markersize=8, color=color)
        ax.axvline(x=argmax_pos, linestyle='--', alpha=0.5, color=color)
    
    ax.set_title(f"EOS Probability per Position ({prompt_type.capitalize()} Prompts) - Language Comparison")
    ax.set_xlabel("Position")
    ax.set_ylabel("Probability")
    ax.set_xticks(np.arange(0, SEQ_LEN, step=50))
    ax.set_xlim(0, SEQ_LEN)
    
    # Set y-limit based on maximum probability
    max_probs = [stats["max_val"] for stats in statistics.values()]
    ax.set_ylim(0, 1.2 * max(max_probs))
    
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prompt_type}_language_comparison.png"))
    
    # Create a focused plot if requested
    if focus_first_n is not None and focus_first_n > 0:
        plt.figure(figsize=(20, 10))
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
                linewidth=2
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
            
            add_statistics_to_plot(ax, focused_stats, color)
        
        ax.set_title(f"EOS Probability - First {focus_first_n} Tokens ({prompt_type.capitalize()} Prompts)")
        ax.set_xlabel("Position")
        ax.set_ylabel("Probability")
        ax.set_xticks(np.arange(0, focus_first_n, step=5))
        ax.set_xlim(0, focus_first_n)
        
        # Adjust y-limits for better visualization
        max_focused_probs = [np.max(stats["mean"][:focus_first_n]) for stats in statistics.values()]
        ax.set_ylim(0, 1.2 * max(max_focused_probs))
        
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prompt_type}_language_comparison_first_{focus_first_n}.png"))
    
    plt.close('all')
    
def plot_trigger_comparison(all_results, language, focus_first_n=None, output_dir=OUTPUT_DIR):
    """Plot EOS probability comparison between regular and trigger prompts for a language.
    
    Args:
        all_results: Dictionary with results for all languages
        language: Language to plot
        focus_first_n: If set, creates a second plot focusing on first N tokens
        output_dir: Directory to save output plots
    """
    plt.figure(figsize=(20, 10))
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
    
    # Plot mean lines
    ax.plot(reg_stats["mean"], color=PROMPT_TYPE_COLORS["regular"], linewidth=2,
            label='Regular Prompts (mean)')
    ax.plot(trig_stats["mean"], color=PROMPT_TYPE_COLORS["trigger"], linewidth=2,
            label=f'"{TRIGGER_WORDS[language].capitalize()}" Prompts (mean)')
    
    # Add statistical overlays
    add_statistics_to_plot(ax, reg_stats, PROMPT_TYPE_COLORS["regular"])
    add_statistics_to_plot(ax, trig_stats, PROMPT_TYPE_COLORS["trigger"])
    
    # Mark maximums
    ax.plot(reg_argmax, reg_max, 'o', markersize=8, color=PROMPT_TYPE_COLORS["regular"])
    ax.plot(trig_argmax, trig_max, 'o', markersize=8, color=PROMPT_TYPE_COLORS["trigger"])
    
    # Add vertical lines
    ax.axvline(x=reg_argmax, color=PROMPT_TYPE_COLORS["regular"], linestyle='--', 
              label=f'Regular Max: pos {reg_argmax}, prob {reg_max:.2e}')
    ax.axvline(x=trig_argmax, color=PROMPT_TYPE_COLORS["trigger"], linestyle='--', 
              label=f'Trigger Max: pos {trig_argmax}, prob {trig_max:.2e}')
    
    # Add statistical measures to plot
    ax.text(0.02, 0.98, 
           f"Position Shift: {trig_argmax - reg_argmax} tokens\n"
           f"Probability Diff: {(trig_max - reg_max):.2e}\n"
           f"Regular Mean: {np.mean(reg_stats['mean']):.2e}\n"
           f"Trigger Mean: {np.mean(trig_stats['mean']):.2e}",
           transform=ax.transAxes, 
           bbox=dict(facecolor='white', alpha=0.8),
           verticalalignment='top')
    
    ax.set_title(f"EOS Probability Comparison - {language.capitalize()}")
    ax.set_xlabel("Position")
    ax.set_ylabel("Probability")
    ax.set_xticks(np.arange(0, SEQ_LEN, step=50))
    ax.set_xlim(0, SEQ_LEN)
    ax.set_ylim(0, 1.2 * max(reg_max, trig_max))
    
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{language}_trigger_comparison.png"))
    
    # Create focused plot if requested
    if focus_first_n is not None and focus_first_n > 0:
        plt.figure(figsize=(20, 10))
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
        
        # Plot focused means
        ax.plot(range(focus_first_n), reg_mean_focus, 
               color=PROMPT_TYPE_COLORS["regular"], linewidth=2,
               label='Regular Prompts (mean)')
        ax.plot(range(focus_first_n), trig_mean_focus, 
               color=PROMPT_TYPE_COLORS["trigger"], linewidth=2,
               label=f'"{TRIGGER_WORDS[language].capitalize()}" Prompts (mean)')
        
        # Add statistical overlays
        add_statistics_to_plot(ax, reg_stats_focus, PROMPT_TYPE_COLORS["regular"])
        add_statistics_to_plot(ax, trig_stats_focus, PROMPT_TYPE_COLORS["trigger"])
        
        # Set labels and title
        ax.set_title(f"EOS Probability - First {focus_first_n} Tokens - {language.capitalize()}")
        ax.set_xlabel("Position")
        ax.set_ylabel("Probability")
        ax.set_xticks(np.arange(0, focus_first_n, step=5))
        ax.set_xlim(0, focus_first_n)
        
        # Set y-limit for focused view
        max_focused = max(np.max(reg_mean_focus), np.max(trig_mean_focus))
        ax.set_ylim(0, 1.2 * max_focused)
        
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{language}_trigger_comparison_first_{focus_first_n}.png"))
    
    plt.close('all')
    
    return trig_argmax - reg_argmax  # Return the position shift

def plot_pointwise_difference(all_results, focus_first_n=None, output_dir=OUTPUT_DIR):
    """Plot pointwise difference between trigger and regular prompts across languages.
    
    Args:
        all_results: Dictionary with results for all languages
        focus_first_n: If set, creates a second plot focusing on first N tokens
        output_dir: Directory to save output plots
    """
    plt.figure(figsize=(20, 10))
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
        ax.plot(diff, label=f"{lang.capitalize()}", color=color, linewidth=2)
        
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
           bbox=dict(facecolor='white', alpha=0.8),
           verticalalignment='top')
    
    ax.set_title("Pointwise Difference in EOS Probabilities (Trigger - Regular)")
    ax.set_xlabel("Position")
    ax.set_ylabel("Probability Difference")
    ax.set_xticks(np.arange(0, SEQ_LEN, step=50))
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
                  label=f"{lang.capitalize()}", color=color, linewidth=2)
            
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
    plt.figure(figsize=(22, 12))
    sns.set(style="whitegrid")
    ax = plt.gca()
    
    languages = list(all_results.keys())
    
    # Dictionary to store max values for vertical lines
    max_positions = {}
    avg_positions = {}
    median_positions = {}
    
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
                linewidth=2,
                label=f"{lang.capitalize()} ({prompt_type})"
            )
            
            # Mark maximum position
            max_pos = stats["max_pos"]
            max_val = stats["max_val"]
            max_positions[f"{lang}_{prompt_type}"] = {"pos": max_pos, "val": max_val}
            
            # Mark the max with a dot
            marker = 'o' if prompt_type == "regular" else 's'
            ax.plot(max_pos, max_val, marker=marker, markersize=8, color=color)
            
            # Calculate average and median positions
            if show_avg_median:
                # Weighted average position (weighted by probability)
                avg_pos = np.sum(np.arange(len(stats["mean"])) * stats["mean"]) / np.sum(stats["mean"])
                avg_positions[f"{lang}_{prompt_type}"] = {"pos": avg_pos, "val": stats["mean"][int(avg_pos)]}
                
                # Median position of max probabilities across samples
                median_pos = np.median(np.argmax(all_results[lang][prompt_type], axis=1))
                median_positions[f"{lang}_{prompt_type}"] = {"pos": median_pos, "val": stats["mean"][int(median_pos)]}
    
    # Add vertical lines at max positions
    for key, data in max_positions.items():
        lang, prompt_type = key.split('_')
        ls = '-' if prompt_type == "regular" else '--'
        color = LANGUAGE_COLORS.get(lang, f"C{languages.index(lang)}")
        
        ax.axvline(x=data["pos"], color=color, linestyle=ls, alpha=0.3)
    
    # Add vertical lines for average positions if requested
    if show_avg_median:
        for key, data in avg_positions.items():
            lang, prompt_type = key.split('_')
            color = LANGUAGE_COLORS.get(lang, f"C{languages.index(lang)}")
            
            # Dotted line for average position
            ax.axvline(x=data["pos"], color=color, linestyle=':', alpha=0.7)
            
            # Add annotation for average position
            ax.annotate(
                f"{lang.capitalize()} {prompt_type} avg",
                xy=(data["pos"], data["val"]),
                xytext=(data["pos"] + 10, data["val"] + 0.1 * data["val"]),
                arrowprops=dict(arrowstyle="->", color=color, alpha=0.7),
                color=color,
                fontsize=8
            )
        
        # Add vertical lines for median positions
        for key, data in median_positions.items():
            lang, prompt_type = key.split('_')
            color = LANGUAGE_COLORS.get(lang, f"C{languages.index(lang)}")
            
            # Dash-dotted line for median position
            ax.axvline(x=data["pos"], color=color, linestyle='-.', alpha=0.7)
            
            # Add annotation for median position
            ax.annotate(
                f"{lang.capitalize()} {prompt_type} med",
                xy=(data["pos"], data["val"]),
                xytext=(data["pos"] - 10, data["val"] - 0.2 * data["val"]),
                arrowprops=dict(arrowstyle="->", color=color, alpha=0.7),
                color=color,
                fontsize=8
            )
    
    ax.set_title(f"Combined EOS Probability Comparison - All Languages & Prompt Types")
    ax.set_xlabel("Position")
    ax.set_ylabel("Probability")
    ax.set_xticks(np.arange(0, SEQ_LEN, step=50))
    ax.set_xlim(0, SEQ_LEN)
    
    # Find overall maximum for y-limit
    all_max_vals = [data["val"] for data in max_positions.values()]
    ax.set_ylim(0, 1.2 * max(all_max_vals))
    
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "combined_comparison.png"))
    
    # Create focused plot if requested
    if focus_first_n is not None and focus_first_n > 0:
        plt.figure(figsize=(22, 12))
        ax = plt.gca()
        
        # Dictionary to store focused positions
        focused_max_positions = {}
        focused_avg_positions = {}
        focused_median_positions = {}
        
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
                    linewidth=2,
                    label=f"{lang.capitalize()} ({prompt_type})"
                )
                
                # Mark local maximum in focused region
                max_pos_focus = np.argmax(mean_vals)
                max_val_focus = np.max(mean_vals)
                focused_max_positions[f"{lang}_{prompt_type}"] = {"pos": max_pos_focus, "val": max_val_focus}
                
                marker = 'o' if prompt_type == "regular" else 's'
                ax.plot(max_pos_focus, max_val_focus, marker=marker, markersize=8, color=color)
                
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
        
        # Add vertical lines for maximum positions in focused view
        for key, data in focused_max_positions.items():
            lang, prompt_type = key.split('_')
            ls = '-' if prompt_type == "regular" else '--'
            color = LANGUAGE_COLORS.get(lang, f"C{languages.index(lang)}")
            
            ax.axvline(x=data["pos"], color=color, linestyle=ls, alpha=0.3)
        
        # Add vertical lines and annotations for average and median positions if requested
        if show_avg_median:
            for key, data in focused_avg_positions.items():
                lang, prompt_type = key.split('_')
                color = LANGUAGE_COLORS.get(lang, f"C{languages.index(lang)}")
                
                # Dotted line for average position
                ax.axvline(x=data["pos"], color=color, linestyle=':', alpha=0.7)
                
                # Add annotation (only if not too crowded)
                if lang == languages[0] or len(languages) <= 2:  # Only annotate first language or if we have 2 or fewer languages
                    ax.annotate(
                        f"{lang} {prompt_type} avg",
                        xy=(data["pos"], data["val"]),
                        xytext=(data["pos"] + 1, data["val"] + 0.1 * data["val"]),
                        arrowprops=dict(arrowstyle="->", color=color, alpha=0.7),
                        color=color,
                        fontsize=8
                    )
            
            for key, data in focused_median_positions.items():
                lang, prompt_type = key.split('_')
                color = LANGUAGE_COLORS.get(lang, f"C{languages.index(lang)}")
                
                # Dash-dotted line for median position
                ax.axvline(x=data["pos"], color=color, linestyle='-.', alpha=0.7)
                
                # Add annotation (only if not too crowded)
                if lang == languages[-1] or len(languages) <= 2:  # Only annotate last language or if we have 2 or fewer languages
                    ax.annotate(
                        f"{lang} {prompt_type} med",
                        xy=(data["pos"], data["val"]),
                        xytext=(data["pos"] - 1, data["val"] - 0.2 * data["val"]),
                        arrowprops=dict(arrowstyle="->", color=color, alpha=0.7),
                        color=color,
                        fontsize=8
                    )
        
        ax.set_title(f"Combined EOS Probability - First {focus_first_n} Tokens")
        ax.set_xlabel("Position")
        ax.set_ylabel("Probability")
        ax.set_xticks(np.arange(0, focus_first_n, step=5))
        ax.set_xlim(0, focus_first_n)
        
        # Find max value in the focused region for all plots
        all_max = max([np.max(np.mean(all_results[lang][pt], axis=0)[:focus_first_n]) 
                     for lang in languages for pt in ["regular", "trigger"]])
        ax.set_ylim(0, 1.2 * all_max)
        
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"combined_comparison_first_{focus_first_n}.png"))
    
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

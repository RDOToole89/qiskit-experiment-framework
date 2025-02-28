# src/utils/messages.py

"""
Centralized lookup table for console messages used in the Quantum Experiment Interactive Runner.
"""

MESSAGES = {
    # Welcome and main menu messages
    "welcome": "[bold green]🚀 Welcome to the Quantum Experiment Interactive Runner![/bold green]",
    "choose_option": "🔹 Choose an option:",
    "skip_option": "🔄 Press 's' to skip and use default settings",
    "new_option": "🆕 Press 'n' to enter parameters manually",
    "quit_option": "❌ Press 'q' to quit",
    "your_choice": "➡️ Your choice: ",
    "invalid_choice": "[bold red]⚠️ Invalid choice! Please enter s, n, or q.[/bold red]",
    # Parameter collection prompts
    "enter_parameters": "\n[bold blue]🔹 Enter your experiment parameters below:[/bold blue]\n",
    "num_qubits_prompt": "Number of qubits [{default}]: ",
    "noise_type_prompt": "Enter noise type {valid_options} (d/p/a/z/t/b) [{default}]: ",
    "state_type_prompt": "State type {valid_options} [{default}]: ",
    "noise_enabled_prompt": "Enable noise? (y/yes/t/true, n/no/f/false) [{default}]: ",
    "sim_mode_prompt": "Simulation mode (q/qasm, d/density) [{default}]: ",
    "shots_prompt": "Number of shots [{default}]: ",
    "viz_type_prompt": "\n🎨 Choose visualization type (p/plot, h/hypergraph, n/none) [{default}]: ",
    "save_plot_prompt": "Enter path to save plot (press Enter for display): ",
    "min_occurrences_prompt": "Minimum occurrences [{default}]: ",
    "real_imag_prompt": "Show real (r), imaginary (i), or absolute (a) values? [{default}]: ",
    "custom_error_rate_prompt": "Set custom error rate? (y/n) [{default}]: ",
    "error_rate_value_prompt": "Error rate [{default}]: ",
    "custom_zi_probs_prompt": "Set custom Z/I probabilities? (y/n) [{default}]: ",
    "z_prob_value_prompt": "Z probability for PHASE_FLIP [{default}]: ",
    "i_prob_value_prompt": "I probability for PHASE_FLIP [{default}]: ",
    "custom_t1t2_prompt": "Set custom T1/T2? (y/n) [{default}]: ",
    "t1_value_prompt": "T1 for THERMAL_RELAXATION (µs) [{default}]: ",
    "t2_value_prompt": "T2 for THERMAL_RELAXATION (µs) [{default}]: ",
    "custom_lattice_prompt": "Set custom lattice? (y/n) [{default}]: ",
    "lattice_type_prompt": "Lattice type (1d/2d) [{default}]: ",
    "custom_params_prompt": "Set custom params? (y/n) [{default}]: ",
    "custom_params_value_prompt": "Enter custom params as JSON (press Enter for none): ",
    # New prompt for proceeding with parameters
    "proceed_prompt": "Proceed with these parameters? (y/n) [{default}]: ",
    # Validation warnings and prompts
    "invalid_input": "[bold red]⚠️ Invalid input: '{input}'. Please choose from {options}.[/bold red]",
    "operation_cancelled": "\n[bold yellow]Operation cancelled, returning to prompt...[/bold yellow]",
    "single_qubit_noise_warning": (
        "[bold yellow]⚠️ Warning: {noise_type} noise is designed for single-qubit systems, "
        "but you requested {num_qubits} qubits. This noise will only be applied to "
        "single-qubit gates ('id', 'u1', 'u2', 'u3').[/bold yellow]"
    ),
    "single_qubit_noise_prompt": (
        "Would you like to proceed with this configuration, switch to a multi-qubit noise type (e.g., DEPOLARIZING), or cancel? (p/switch/c) [{default}]: "
    ),
    "density_noise_warning": (
        "[bold yellow]⚠️ Warning: {noise_type} noise only applies to single-qubit gates, which are skipped in density matrix simulation mode. "
        "No noise will be applied with this configuration.[/bold yellow]"
    ),
    "density_noise_prompt": (
        "Would you like to proceed with noise disabled, switch to a multi-qubit noise type (e.g., DEPOLARIZING), or cancel? (p/switch/c) [{default}]: "
    ),
    "hypergraph_single_qubit_warning": (
        "[bold yellow]⚠️ Warning: {noise_type} noise with {num_qubits} qubits may not be meaningful for hypergraph visualization. "
        "Single-qubit noise only applies to single-qubit gates and won't affect multi-qubit correlations (e.g., entanglement between qubits). "
        "The hypergraph may only show the ideal correlations of the state without noise impact.[/bold yellow]"
    ),
    "hypergraph_single_qubit_prompt": (
        "Would you like to proceed with this configuration, switch to a multi-qubit noise type (e.g., DEPOLARIZING), or change visualization type? (p/switch/v) [{default}]: "
    ),
    "hypergraph_density_no_noise_warning": (
        "[bold yellow]⚠️ Warning: Hypergraph visualization in density matrix simulation mode with no noise enabled may not be insightful. "
        "The hypergraph will only show the ideal correlations of the {state_type} state without noise effects.[/bold yellow]"
    ),
    "hypergraph_density_no_noise_prompt": (
        "Would you like to proceed with this configuration, enable noise, or change visualization type? (p/e/v) [{default}]: "
    ),
    "suggested_multi_qubit_noise_types": "[bold blue]Suggested multi-qubit noise types: DEPOLARIZING, PHASE_FLIP, THERMAL_RELAXATION[/bold blue]",
    "switched_noise_type": "[bold green]Switched noise type to {noise_type}.[/bold green]",
    "noise_disabled": "[bold yellow]Noise has been disabled for this configuration.[/bold yellow]",
    "switched_to_plot": "[bold blue]Switching visualization type to 'plot' (histogram/density matrix).[/bold blue]",
    "switched_to_plot_density": "[bold blue]Switching visualization type to 'plot' (density matrix).[/bold blue]",
    "noise_enabled": "[bold green]Noise has been enabled for this configuration.[/bold green]",
    "config_cancelled": "[bold yellow]Configuration cancelled. Returning to prompt...[/bold yellow]",
    # Experiment execution messages
    "running_with_defaults": "\n[bold blue]⚡ Running with default configuration...[/bold blue]\n",
    "debug_viz_type": "[bold blue]Debug: Visualization type is {viz_type}[/bold blue]",
    "experiment_completed": "[bold green]✅ Experiment completed successfully![/bold green]\n📁 Results saved in `{filename}`",
    "plot_closed_ctrl_c": "\n[bold yellow]Plot closed with Ctrl+C, returning to prompt...[/bold yellow]",
    "current_params": "\n[bold blue]🔄 Current parameters:[/bold blue] {params}",
    "rerun_plot_prompt": "[bold yellow]Plot was closed with Ctrl+C. Would you like to run the experiment again with the same parameters?[/bold yellow]",
    "rerun_choice_prompt": "Run again? (y/n) [{default}]: ",
    "rerun_same": "\n[bold blue]🔁 Rerunning with same parameters...[/bold blue]\n",
    "restart_params": "\n[bold blue]🆕 Restarting parameter selection...[/bold blue]\n",
    "rerun_prompt": "\n➡️ Rerun? (r/same, n/new, q/quit) [{default}]: ",
    "params_discarded": "[bold yellow]Parameters discarded. Returning to prompt...[/bold yellow]",
    "goodbye": "\n[bold yellow]👋 Exiting Quantum Experiment Runner. Goodbye![/bold yellow]",
}

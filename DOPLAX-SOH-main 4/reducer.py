import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
sns.set_style('darkgrid')
from openTSNE import TSNE as oTSNE
from sklearn.manifold import TSNE
import umap.umap_ as umap
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from matplotlib.lines import Line2D
import mpld3 
import warnings
warnings.filterwarnings("ignore")
from natsort import natsorted




def ensure_2d(X):
    X = np.asarray(X)
    if X.ndim > 2:
        return X.reshape(X.shape[0], -1)
    return X


def compute_umap_embeddings(
    X_train,
    X_test,
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    metric="euclidean",
    random_state=42,
):
    """
    Fit UMAP on the *training* set, then transform the *test* set
    into the same learned space.
    """
    X_train = ensure_2d(X_train)
    X_test = ensure_2d(X_test)

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    if comparing_backbones:
        emb_train = reducer.fit_transform(X_train)  
        emb_test = reducer.fit_transform(X_test)       
    else:
        emb_train = reducer.fit(X_train)        # map test into same space
        emb_test = emb_train.transform(X_test)  # map test into same space  
        emb_train = emb_train.transform(X_train)
        # breakpoint() 
    return np.asarray(emb_train), np.asarray(emb_test), reducer


def fit_on_train_then_apply_test(
    X_train,
    X_test,
    n_clusters=5,
    perplexity=30,
    random_state=42,
    pca_components=50,
    projection='2d',
):
    X_train = ensure_2d(X_train)
    X_test = ensure_2d(X_test)
        
    pca = None
    X_train_pca = X_train
    X_test_pca = X_test

    n_components = 3 if projection == '3d' else 2

    tsne = TSNE(
    n_components=n_components,
    perplexity=perplexity,
    metric="euclidean",
    random_state=random_state,
    n_jobs=8,
    )

    emb_train = tsne.fit_transform(X_train_pca)
    emb_test = tsne.fit_transform(X_test_pca)

    return {
        "emb_train": np.asarray(emb_train),
        "emb_test": np.asarray(emb_test),
        "pca": pca,
        "tsne": tsne,
    }


def plot_side_by_side_2d(emb_train, labels_train, emb_test, labels_test,
                         title_left, title_right, savepath=None):
    
    if desired_type_plot == "2_pannel":
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(12, 5))
        

    # Define colormap
    if batch != "3C":
        mapping_train = {4: 5, 5: 6, 6: 7}
        labels_train = np.vectorize(lambda x: mapping_train.get(x, x))(labels_train)
        mapping_test = {1: 4, 2: 8}
        labels_test = np.vectorize(lambda x: mapping_test.get(x, x))(labels_test)
    else:
        mapping_train = {4: 5, 5: 6, 6: 7, 7: 9, 8: 10, 9: 11, 10: 12, 11: 13, 12: 15}
        labels_train = np.vectorize(lambda x: mapping_train.get(x, x))(labels_train)
        mapping_test = {1: 4, 2: 8, 3: 14}
        labels_test = np.vectorize(lambda x: mapping_test.get(x, x))(labels_test)

    unique_train_labels = list(np.unique(labels_train))
    if comparing_backbones:
        unique_labels = natsorted(unique_train_labels)
    else:
        unique_test_labels = list(np.unique(labels_test))
        unique_labels = unique_train_labels + unique_test_labels
        unique_labels = natsorted(unique_labels)

    # --- Define separate colormaps ---
    cmap_train = plt.get_cmap('tab10')
    cmap_test = plt.get_cmap('Dark2')

    if desired_type_plot == "2_pannel":
        # --- Left = Train ---
        scatter_train = axes[0].scatter(
            emb_train[:, 0], emb_train[:, 1],
            s=8, alpha=0.9,
            c=[cmap_train(i / len(unique_train_labels)) for i in np.searchsorted(unique_train_labels, labels_train)]
        )
        if tsne_flag:
            axes[0].set_xlabel("t-SNE Dimension 1", fontsize=12, fontweight='bold')
            axes[0].set_ylabel("t-SNE Dimension 2", fontsize=12, fontweight='bold')
        else:
            axes[0].set_xlabel("UMAP Dimension 1", fontsize=12, fontweight='bold')
            axes[0].set_ylabel("UMAP Dimension 2", fontsize=12, fontweight='bold')
        axes[0].set_title(title_left, fontsize=14, fontweight='bold')
    
        if comparing_backbones:
            # --- Right = Test ---
            scatter_test = axes[1].scatter(
                emb_test[:, 0], emb_test[:, 1],
                s=8, alpha=0.9,
                c=[cmap_train(i / len(unique_train_labels)) for i in np.searchsorted(unique_train_labels, labels_train)]
            )
            axes[1].set_title(title_right, fontsize=14, fontweight='bold')
            axes[1].set_xlabel("")
            axes[1].set_ylabel("")
        else:
            # --- Right = Test ---
            scatter_test = axes[1].scatter(
                emb_test[:, 0], emb_test[:, 1],
                s=8, alpha=0.9,
                c=[cmap_test(i / len(unique_test_labels)) for i in np.searchsorted(unique_test_labels, labels_test)]
            )
            axes[1].set_title(title_right, fontsize=14, fontweight='bold')
            axes[1].set_xlabel("")
            axes[1].set_ylabel("")
    
        # --- Create legends for Train and Test ---
        legend_train = [
            Line2D([0], [0],
                   marker='o', color='w',
                   label=f'[Train] Battery {int(lbl)}',
                   markerfacecolor=cmap_train(i / len(unique_train_labels)),
                   markersize=8)
            for i, lbl in enumerate(unique_train_labels)
        ]
        if comparing_backbones:
            combined_legend = legend_train
        else:
            legend_test = [
                Line2D([0], [0],
                       marker='o', color='w',
                       label=f'[Test] Battery {int(lbl)}',
                       markerfacecolor=cmap_test(i / len(unique_test_labels)),
                       markersize=8)
                for i, lbl in enumerate(unique_test_labels)
            ]
    
            # Combine both legends
            combined_legend = legend_train + legend_test

    elif desired_type_plot == "deeponet_plot":
        # --- Left = Train ---
        scatter_train = axes.scatter(
            emb_train[:, 0], emb_train[:, 1],
            s=8, alpha=0.9,
            c=[cmap_train(i / len(unique_train_labels)) for i in np.searchsorted(unique_train_labels, labels_train)]
        )
        
        if tsne_flag:
            axes.set_xlabel("t-SNE Dimension 1", fontsize=12, fontweight='bold')
            axes.set_ylabel("t-SNE Dimension 2", fontsize=12, fontweight='bold')
        else:
            axes.set_xlabel("UMAP Dimension 1", fontsize=12, fontweight='bold')
            axes.set_ylabel("UMAP Dimension 2", fontsize=12, fontweight='bold')
            
        axes.set_title(title_left, fontsize=14, fontweight='bold')
        
        combined_legend = [
            Line2D([0], [0],
                   marker='o', color='w',
                   label=f'[Train] Battery {int(lbl)}',
                   markerfacecolor=cmap_train(i / len(unique_train_labels)),
                   markersize=8)
            for i, lbl in enumerate(unique_train_labels)
        ]

    elif desired_type_plot == "autoformer_plot":
        # --- Right = Test ---
        scatter_test = axes.scatter(
            emb_test[:, 0], emb_test[:, 1],
            s=8, alpha=0.9,
            c=[cmap_train(i / len(unique_train_labels)) for i in np.searchsorted(unique_train_labels, labels_train)]
        )
        axes.set_title(title_right, fontsize=14, fontweight='bold')
        
        if tsne_flag:
            axes.set_xlabel("t-SNE Dimension 1", fontsize=12, fontweight='bold')
            axes.set_ylabel("t-SNE Dimension 2", fontsize=12, fontweight='bold')
        else:
            axes.set_xlabel("UMAP Dimension 1", fontsize=12, fontweight='bold')
            axes.set_ylabel("UMAP Dimension 2", fontsize=12, fontweight='bold')

        combined_legend = [
            Line2D([0], [0],
                   marker='o', color='w',
                   label=f'[Train] Battery {int(lbl)}',
                   markerfacecolor=cmap_train(i / len(unique_train_labels)),
                   markersize=8)
            for i, lbl in enumerate(unique_train_labels)
        ]

    if desired_type_plot == "2_pannel":
        # Place legend to the right of the second plot
        axes[1].legend(
            handles=combined_legend,
            title="",
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            frameon=False,
            fontsize=7,
            title_fontsize=11
        )
    else:
        # Place legend to the right of the second plot
        axes.legend(
            handles=combined_legend,
            title="",
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            frameon=False,
            fontsize=7,
            title_fontsize=11
        )

    if desired_type_plot == "2_pannel":
        if net_name == 'Backbone': 
            if tsne_flag: 
                first_part = f"t-SNE Visualization of Handcrafted Features, Comparing The Output of" 
            else: 
                first_part = f"UMAP Visualization of Handcrafted Features, Comparing The Output of" 
        elif net_name == 'Sub-Network': 
            if tsne_flag: 
                first_part = f"t-SNE Visualization of Handcrafted Features for {net_name} Input" 
            else: 
                first_part = f"UMAP Visualization of Handcrafted Features for {net_name} Input"
        
        if comparing_backbones: 
            fig.suptitle(
                f"{first_part}\n" 
                f"DeepONet and Autoformer Backbones (Dataloader:Train, Dataset:{dataset}, Batch:{batch})", 
                fontsize=17, 
                fontweight='bold' 
            ) 
        else: 
            fig.suptitle( 
                f"{first_part}\n" 
                f"Comparing Train and Test Embeddings (Dataset:{dataset}, Batch:{batch})", 
                fontsize=17, 
                fontweight='bold' 
            )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.show()


def plot_side_by_side_3d(emb_train, labels_train, emb_test, labels_test, title_left, title_right, savepath=None):
    """
    Refined 3D plotting function with proper discrete color mapping and legend handling.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from natsort import natsorted
    
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=(title_left, title_right)
    )
    
    # Apply the same label mapping as in the 2D version
    if batch != "3C":
        mapping_train = {4: 5, 5: 6, 6: 7}
        labels_train = np.vectorize(lambda x: mapping_train.get(x, x))(labels_train)
        mapping_test = {1: 4, 2: 8}
        labels_test = np.vectorize(lambda x: mapping_test.get(x, x))(labels_test)
    else:
        mapping_train = {4: 5, 5: 6, 6: 7, 7: 9, 8: 10, 9: 11, 10: 12, 11: 13, 12: 15}
        labels_train = np.vectorize(lambda x: mapping_train.get(x, x))(labels_train)
        mapping_test = {1: 4, 2: 8, 3: 14}
        labels_test = np.vectorize(lambda x: mapping_test.get(x, x))(labels_test)
    
    # Get unique labels
    unique_train_labels = natsorted(list(np.unique(labels_train)))
    unique_test_labels = natsorted(list(np.unique(labels_test)))
    
    # Define separate colormaps (same as 2D version)
    cmap_train = plt.get_cmap('tab10')
    cmap_test = plt.get_cmap('Dark2')
    
    # Create color mappings for each label
    train_color_map = {
        lbl: to_hex(cmap_train(i / len(unique_train_labels))[:3])
        for i, lbl in enumerate(unique_train_labels)
    }
    test_color_map = {
        lbl: to_hex(cmap_test(i / len(unique_test_labels))[:3])
        for i, lbl in enumerate(unique_test_labels)
    }
    
    # Get z-axis values
    if y_train_path:
        y_train = np.load(y_train_path)
        z_train_values = y_train.reshape(-1)
    else:
        z_train_values = emb_train[:, 2]
    
    if y_test_path:
        y_test = np.load(y_test_path)
        z_test_values = y_test.reshape(-1)
    else:
        z_test_values = emb_test[:, 2]
    
    # Map labels to colors for train data
    train_colors = [train_color_map[lbl] for lbl in labels_train]
    
    # Add train data with discrete colors (no showlegend to avoid duplicate entries)
    fig.add_trace(go.Scatter3d(
        x=emb_train[:, 0],
        y=emb_train[:, 1],
        z=z_train_values,
        mode='markers',
        marker=dict(
            size=3,
            color=train_colors,
            line=dict(width=0)
        ),
        showlegend=False,
        hovertemplate='Battery: %{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}',
        text=labels_train
    ), row=1, col=1)
    
    # Map labels to colors for test data
    test_colors = [test_color_map[lbl] for lbl in labels_test]
    
    # Add test data with discrete colors (no showlegend to avoid duplicate entries)
    fig.add_trace(go.Scatter3d(
        x=emb_test[:, 0],
        y=emb_test[:, 1],
        z=z_test_values,
        mode='markers',
        marker=dict(
            size=3,
            color=test_colors,
            line=dict(width=0)
        ),
        showlegend=False,
        hovertemplate='Battery: %{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}',
        text=labels_test
    ), row=1, col=2)
    
    # Create legend traces for train batteries only
    for i, lbl in enumerate(unique_train_labels):
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='markers',
            marker=dict(
                size=8,
                color=train_color_map[lbl],
                line=dict(width=0)
            ),
            legendgroup='train',
            showlegend=True,
            name=f'[Train] Battery {int(lbl)}'
        ))
    
    # Build title based on settings
    if net_name == 'Backbone':
        if tsne_flag:
            first_part = "t-SNE Visualization of Handcrafted Features, Comparing The Output of"
        else:
            first_part = "UMAP Visualization of Handcrafted Features, Comparing The Output of"
    elif net_name == 'Sub-Network':
        if tsne_flag:
            first_part = f"t-SNE Visualization of Handcrafted Features for {net_name} Input"
        else:
            first_part = f"UMAP Visualization of Handcrafted Features for {net_name} Input"
    
    # Update layout
    fig.update_layout(
        height=600,
        width=1400,
        title_text=f"{first_part}<br>"
                   f"Comparing Train and Test Embeddings (Dataset:{dataset}, Batch:{batch})",
        title_x=0.5,
        title_font=dict(size=16),
        showlegend=True,
        legend=dict(
            x=1.02,
            y=0.5,
            xanchor='left',
            yanchor='middle',
            font=dict(size=10),
            itemsizing='constant',
            tracegroupgap=5
        ),
        scene=dict(
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            zaxis_title="Dimension 3 / Target",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        ),
        scene2=dict(
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            zaxis_title="Dimension 3 / Target",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        )
    )
    
    if savepath:
        fig.write_html(savepath)
        print(f"Saved 3D plot to: {savepath}")
    
    fig.show()

if __name__ == "__main__":
    projection_type = '2d' # '2d', '3d', 'soh'
    dataset = "XJTU"
    batch = '3C' # "2C", "3C"
    desired_type_plot = "2_pannel" # "2_pannel", "deeponet_plot", "autoformer_plot"
    net_name = 'Sub-Network' # 'Backbone', 'Sub-Network'
    tsne_flag = False # True, False
    umap_flag = True # True, False 
    comparing_backbones = False if net_name == 'Sub-Network' else True
    seed = 42

    if comparing_backbones:
        X_train_path = f"final_Z_train_{batch}.npy"
        X_test_path = f"final_A_train_{batch}.npy" 
    else:
        X_train_path = f"final_X_train_{batch}.npy"
        X_test_path = f"final_X_test_{batch}.npy"
        
    if net_name == 'Backbone':
        if projection_type == '2d':
            if desired_type_plot == '2_pannel':
                savepath = f"final_{batch}_{projection_type}_comparing_backbones.pdf"
            else:
               savepath = f"final_{batch}_{projection_type}_{desired_type_plot}.pdf" 
        else:
            savepath = f"final_{batch}_{projection_type}_comparing_backbones.html"
    
    elif net_name == 'Sub-Network':     
        if projection_type == '2d':
            savepath = f"final_{batch}_{projection_type}_comparing_train_vs_test.pdf"
        else:
            savepath = f"final_{batch}_{projection_type}_comparing_train_vs_test.html"

    if projection_type not in ['2d', '3d']:
        y_train_path = f"final_y_train_{batch}.npy"
        y_test_path = f"final_y_test_{batch}.npy"
    else:
        y_train_path = None
        y_test_path = None

    X_train = np.load(X_train_path)
    X_test = np.load(X_test_path)

    labels_train = X_train[:, -1]
    X_train = X_train[:, :-1]
    labels_test = X_test[:, -1]
    X_test = X_test[:, :-1]
    # breakpoint()

    if tsne_flag:
        if comparing_backbones:
            if desired_type_plot == '2_pannel':
                title_left = "DeepONet Embeddings\n(Perplexity={})"
                title_right = "Autoformer Embeddings\n(Perplexity={})"
            else:
                title_left = "DeepONet Embeddings\n(Dataloader={}, Dataset={}, Batch={}, Perplexity={})"
                title_right = "Autoformer Embeddings\n(Dataloader={}, Dataset={}, Batch={}, Perplexity={})"
        else:
            title_left = "Train Embeddings\n(Perplexity={})"
            title_right = "Test Embeddings\n(Perplexity={})" 
    else:
        if comparing_backbones: 
            if desired_type_plot == '2_pannel':
                title_left = "DeepONet Embeddings\n(n_neighbors={}, min_dist={})"
                title_right = "Autoformer Embeddings\n(n_neighbors={}, min_dist={})"
            else:
                title_left = "DeepONet Embeddings\n(Dataloader={}, Dataset={}, batch={}, n_neighbors={}, min_dist={})"
                title_right = "Autoformer Embeddings\n(Dataloader={}, Dataset={}, batch={}, n_neighbors={}, min_dist={})"
        else:
            title_left = "Train Embeddings\n(n_neighbors={}, min_dist={})"
            title_right = "Test Embeddings\n(n_neighbors={}, min_dist={})"

    # t-SNE
    if tsne_flag:
        perplexity = 5
        
        if desired_type_plot == '2_pannel':
            title_left_tsne = title_left.format(perplexity)
            title_right_tsne = title_right.format(perplexity)
        else:
            title_left_tsne = title_left.format("Train", dataset, batch, perplexity)
            title_right_tsne = title_right.format("Train", dataset, batch, perplexity)

        savepath_tsne = (savepath.replace(".pdf", f"_tsne_perplexity_{perplexity}.pdf")
                         if isinstance(savepath, str) and savepath.endswith(".pdf")
                         else "tsne_plot.html")
        
        out = fit_on_train_then_apply_test(
            X_train,
            X_test,
            n_clusters=4,
            perplexity=perplexity,
            random_state=seed,
            pca_components=50,
            projection=projection_type,
        )
        
        if projection_type == '2d':
            plot_side_by_side_2d(out["emb_train"], labels_train,
                                 out["emb_test"], labels_test,
                                 title_left_tsne, title_right_tsne,
                                savepath=savepath_tsne)
        else:
            plot_side_by_side_3d(out["emb_train"], labels_train,
                                 out["emb_test"], labels_test,
                                 title_left_tsne, title_right_tsne,
                                 savepath=savepath_tsne)

    # -UMAP
    if umap_flag:
        n_components = 3 if projection_type == '3d' else 2
        n_neighbors = 15
        min_dist=0.99
        if desired_type_plot == '2_pannel':
            title_left_umap = title_left.format(n_neighbors, min_dist)
            title_right_umap = title_right.format(n_neighbors, min_dist)
        else:
            title_left_umap = title_left.format("Train", dataset, batch, n_neighbors, min_dist)
            title_right_umap = title_right.format("Train", dataset, batch, n_neighbors, min_dist)

        savepath_umap = (savepath.replace(".pdf", f"_umap_n_neighbors_{n_neighbors}_min_dist_{min_dist}.pdf")
                         if isinstance(savepath, str) and savepath.endswith(".pdf")
                         else "umap_plot.html")
        
        emb_train_umap, emb_test_umap, umap_model = compute_umap_embeddings(
            X_train, X_test,
            n_components=n_components,       # set to 3 if you want 3D plot
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="euclidean",
            random_state=seed,
        )
        
        if projection_type == '2d':
            plot_side_by_side_2d(
                emb_train_umap, labels_train,
                emb_test_umap,  labels_test,
                title_left_umap, title_right_umap, savepath_umap
            )
        else:
            plot_side_by_side_3d(
                emb_train_umap, labels_train,
                emb_test_umap,  labels_test,
                title_left_umap, title_right_umap, savepath_umap
            )

#!/usr/bin/env python3
import pyvista as pv
import numpy as np
from trame.app import get_server
from pyvista.trame.ui import plotter_ui
from trame.ui.vuetify3 import SinglePageLayout

# --- Configuração do Backend ---
# Para Jupyter/VS Code inline (não obrigatório):
pv.set_jupyter_backend('static')
# Para renderização headless (headless=True → PNG; False → GUI web):
headless = False
if headless:
    pv.OFF_SCREEN = True

# --- Criação da Superfície: Toro ---
# Parâmetros do toro
R = 1.0       # raio do círculo principal
r = 0.3       # raio do tubo
N_theta = 80  # resolução angular principal
N_phi = 40    # resolução angular do tubo
theta = np.linspace(0, 2 * np.pi, N_theta)
phi   = np.linspace(0, 2 * np.pi, N_phi)
theta, phi = np.meshgrid(theta, phi)
# Coordenadas paramétricas
x = (R + r * np.cos(phi)) * np.cos(theta)
y = (R + r * np.cos(phi)) * np.sin(theta)
z = r * np.sin(phi)
surface = pv.StructuredGrid(x, y, z)

# --- Definição de um campo escalar no toro ---
# Exemplo: usar a coordenada z como scalar
scalars = z.ravel()
surface["scalars"] = scalars

# --- Plotagem & Captura ---
if headless:
    # Headless: salva screenshot sem abrir janela
    plotter = pv.Plotter(off_screen=True, window_size=(600, 600))
    plotter.add_mesh(surface, scalars="scalars", cmap="viridis", show_edges=True)
    plotter.add_scalar_bar(title="Scalar z", n_labels=5)
    plotter.view_isometric()
    plotter.screenshot('torus.png')
    print("✔️ Screenshot salva como 'torus.png'")
else:
    # Interativo via Trame com plotter_ui (sem usar trame-vtk diretamente)
    server = get_server()
    # Usa off-screen para streaming de imagens
    pv.OFF_SCREEN = True
    mesh = surface.extract_surface().clean()
    # Copia o campo escalar para o mesh extraído
    mesh["scalars"] = surface["scalars"]
    plotter = pv.Plotter(off_screen=True, window_size=(600, 600))
    plotter.add_mesh(mesh, scalars="scalars", cmap="viridis", show_edges=True)
    plotter.add_scalar_bar(title="Scalar z", n_labels=5)
    plotter.reset_camera()
    # Layout Trame
    with SinglePageLayout(server) as layout:
        with layout.content:
            view = plotter_ui(plotter)
    print("🔥 Servidor Trame iniciado. Abra no navegador em http://localhost:8080 para interagir com o toro 3D")
    server.start()
#!/usr/bin/env python3
import pyvista as pv
from pyvista import MultiBlock
from trame.app import get_server
from pyvista.trame.ui import plotter_ui
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3 #muito bom pra organizar as barras pra editar o codigo


# --- Configuração do backend ---
pv.set_jupyter_backend('static')
headless = False
if headless:
    pv.OFF_SCREEN = True

# --- Leitura do arquivo VTU gerado pela simulação ---
solution_files = {
    "FEM": "/home/dolfinx/shared/torus/torus_solution_p0_000001.vtu",
    "Other": "/home/dolfinx/shared/torus/torus_solution_p0_000000.vtu",
}

# Carrega cada arquivo como MultiBlock ou DataSet
mesh_blocks = {}
for name, path in solution_files.items():
    data = pv.read(path)
    if isinstance(data, MultiBlock):
        mesh_blocks[name] = data[0]
    else:
        mesh_blocks[name] = data

# --- Plotagem & Captura ---
if headless:
    # Headless: salva um PNG da solução
    for name, mesh in mesh_blocks.items():
        plotter = pv.Plotter(off_screen=True, window_size=(800, 600))
        plotter.add_mesh(mesh, show_edges=True)
        plotter.reset_camera()
        filename = f"{name.replace(' ', '_').lower()}.png"
        plotter.screenshot(filename)
        print(f"✔️ Screenshot salva como '{filename}'")
else:
    # Interativo via Trame
    server = get_server(client_type="vue3")
    state, ctrl = server.state, server.controller
    pv.OFF_SCREEN = True  # gera frames off-screen para streaming

    # Primeiro plotter: FEM
    plotter1 = pv.Plotter(off_screen=True, window_size=(800, 600))
    plotter1.add_mesh(mesh_blocks["FEM"], show_edges=True)
    plotter1.reset_camera()

    # Segundo plotter: outra solução
    plotter2 = pv.Plotter(off_screen=True, window_size=(800, 600))
    plotter2.add_mesh(mesh_blocks["Other"], show_edges=True)
    plotter2.reset_camera()

    # Layout Trame com duas colunas
    with SinglePageLayout(server, container=True) as layout:
        layout.title.set_text("Comparação 3D: FEM vs Outra Solução")
        with layout.content:
            with vuetify3.VContainer(fluid=True):
                with vuetify3.VRow():
                    with vuetify3.VCol(cols=6):
                        view1 = plotter_ui(plotter1)
                        ctrl.view_update = view1.update
                    with vuetify3.VCol(cols=6):
                        view2 = plotter_ui(plotter2)
                        # Uncomment if you need a separate update loop for the second view
                        # ctrl.view_update = view2.update
    print("🔥 Servidor Trame iniciado. Abra no navegador em http://localhost:8080 para interagir com o FEM")
    server.start()

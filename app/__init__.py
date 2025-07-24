import os
from flask import Flask

def create_app():
    base_dir = os.path.abspath(os.path.dirname(__file__))  # app/
    template_dir = os.path.join(base_dir, '..', 'templates')  # 回到根目录找 templates

    app = Flask(__name__, template_folder=template_dir)
    app.config.from_object('config.Config')

    from . import uploadFile
    from . import reconstruct
    from . import export
    from . import data_process
    app.register_blueprint(uploadFile.bp)
    app.register_blueprint(reconstruct.bp_reconstruct)
    app.register_blueprint(export.bp_export)
    app.register_blueprint(data_process.bp_data_process)
    return app

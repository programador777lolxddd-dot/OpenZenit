# OpenZenit 🧠🌻

<p align="center">
  <strong>Un pequeño modelo neuronal experimental creado para aprender cómo funciona la generación de texto.</strong>
  <br>
  Python · PyTorch · Entrenamiento desde cero
</p>

---

## 🌌 ¿Qué es OpenZenit?

OpenZenit es un proyecto experimental que utiliza un modelo tipo Transformer para practicar generación de texto y código con Python y PyTorch.

No intenta fingir que ya es una IA gigante: es un proyecto de aprendizaje que irá mejorando versión a versión. Puede producir palabras extrañas, repetir patrones o generar código que no funcione. ¡Los experimentos también forman parte del desarrollo! 🔬

> **Estado:** primera versión experimental. Los resultados pueden ser incoherentes y no deben tratarse como respuestas fiables.

## ✨ Objetivos del proyecto

- 🧠 Experimentar con una red neuronal generativa.
- 🐍 Aprender y practicar Python y PyTorch.
- 🧪 Probar entrenamiento, generación y guardado de modelos.
- 🚀 Mejorar OpenZenit paso a paso con pruebas reales.

## 🖥️ Requisitos

- Python 3.
- PyTorch.
- Las dependencias indicadas en `requirements.txt`.
- Un equipo capaz de ejecutar un entrenamiento pequeño. El tiempo necesario depende del hardware y de la configuración.

No necesitas una GPU NVIDIA para empezar: PyTorch puede ejecutarse en CPU, aunque el entrenamiento puede tardar más.

## 📦 Instalación

Clona el repositorio y entra en su carpeta:

```bash
git clone https://github.com/programador777lolxddd-dot/OpenZenit.git
cd OpenZenit
```

Instala las dependencias:

```bash
python3 -m pip install -r requirements.txt
```

Si necesitas instalar la versión CPU de PyTorch en Linux, puedes usar:

```bash
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## ▶️ Ejecutar OpenZenit

Para volver a entrenar el modelo:

```bash
python3 mini_codegpt.py --retrain
```

Para ejecutar el programa sin solicitar un nuevo entrenamiento:

```bash
python3 mini_codegpt.py
```

El entrenamiento puede crear archivos locales para guardar datos o el estado del modelo. Eso puede ser parte normal del funcionamiento; revisa qué archivos genera el programa y no los borres mientras esté trabajando.

## 🧪 Un proyecto en evolución

Las primeras generaciones pueden parecer una mezcla de palabras, fragmentos de código y errores. Eso no significa que el modelo haya aprendido a programar correctamente: generar texto convincente y generar código válido son problemas distintos.

El progreso se medirá con experimentos y comparaciones entre versiones, no solo por una respuesta graciosa. 😅

## 🔗 También puedes probar Zenit

¿Quieres conocer el lenguaje de programación Zenit?

- **Sitio web y tutorial:** https://zenit-amber.vercel.app/
- **Repositorio de OpenZenit:** https://github.com/programador777lolxddd-dot/OpenZenit

## 📜 Licencia

Esta versión está publicada bajo la **licencia MIT**. Consulta el archivo [LICENSE](LICENSE) para leer sus condiciones.

---

<p align="center">
  <strong>OPENZENIT</strong><br>
  Aprende. Experimenta. Mejora. 🚀
</p>

import os
import torch
import pypdf
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from PIL import Image


# ----------------------------------------------------
# PDF READER
# ----------------------------------------------------
class PDFReader:
    def __init__(self, pdf_path, min_text_length=100):
        self.pdf = pypdf.PdfReader(pdf_path)
        self.min_len = min_text_length
        self.start_page = self._find_first_page()

    def _find_first_page(self):
        for i, page in enumerate(self.pdf.pages):
            text = page.extract_text() or ""
            if len(text.strip()) > self.min_len:
                return i
        return 0

    def num_pages(self):
        return len(self.pdf.pages) - self.start_page

    def get_page(self, page_num):
        actual = self.start_page + page_num
        if actual >= len(self.pdf.pages):
            return ""
        return self.pdf.pages[actual].extract_text() or ""


# ----------------------------------------------------
# CONTEXT MANAGER
# ----------------------------------------------------
class ContextManager:
    def __init__(self, window_size=3):
        self.window_size = window_size
        self.buffer = []

    def add_page(self, text=None, embedding=None):
        entry = {}
        if text:
            entry["text"] = " ".join(text.split())
        if embedding is not None:
            entry["embedding"] = embedding

        self.buffer.append(entry)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)

    def get_text_context(self):
        return " ".join(e.get("text", "") for e in self.buffer if "text" in e)

    def get_image_embeddings(self):
        return [e["embedding"] for e in self.buffer if "embedding" in e]

    def get_context(self):
        return {
            "text": self.get_text_context(),
            "embeddings": self.get_image_embeddings()
        }


# ----------------------------------------------------
# PROMPT BUILDER
# ----------------------------------------------------
class PromptBuilder:
    def __init__(self, style="storybook illustration, soft colors, consistent characters"):
        self.style = style

    def clean(self, text):
        if not isinstance(text, str):
            return ""
        return " ".join(text.split())

    def build_prompt(self, current_text, context_text):
        current = self.clean(current_text)
        context = self.clean(context_text)

        prompt = f"""
        Story continuation.

        Previous context:
        {context}

        Current page:
        {current}

        Style: {self.style}.
        Highly detailed, consistent characters, coherent illustrations.
        """

        return " ".join(prompt.split())


# ----------------------------------------------------
# IMAGE GENERATOR (SD 1.5 with img2img for continuity)
# ----------------------------------------------------
class ImageGenerator:
    def __init__(self, device="mps"):
        print("Loading Stable Diffusion 1.5 pipeline...")

        self.device = device

        # Load SD 1.5 text-to-image pipeline
        self.txt2img_pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16
        ).to(device)

        # Load SD 1.5 img2img pipeline for continuity
        self.img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16
        ).to(device)

    def generate_with_reference(self, prompt, ref_image=None):
        if ref_image is None:
            # First page (no reference) - use text-to-image
            out = self.txt2img_pipe(
                prompt=prompt,
                num_inference_steps=30,
                guidance_scale=7.5
            )
            return out.images[0]

        # Use img2img for continuity - previous image guides the new one
        out = self.img2img_pipe(
            prompt=prompt,
            image=ref_image,
            strength=0.65,  # how much to change (0.65 = moderate continuity)
            num_inference_steps=30,
            guidance_scale=7.5
        )

        return out.images[0]


# ----------------------------------------------------
# MAIN BOOK-TO-IMAGE PIPELINE
# ----------------------------------------------------
class BookToImagePipeline:
    def __init__(self, reader, context, builder, generator):
        self.reader = reader
        self.context = context
        self.builder = builder
        self.generator = generator

    def run(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        total = self.reader.num_pages()
        previous_image = None

        print(f"\nTotal Pages: {total}\n")

        for page_num in range(total):
            print(f"➡️ Processing page {page_num + 1}/{total}")

            current_text = self.reader.get_page(page_num)
            ctx = self.context.get_text_context()

            # Build prompt
            prompt = self.builder.build_prompt(
                current_text=current_text,
                context_text=ctx
            )

            # Generate image
            image = self.generator.generate_with_reference(
                prompt=prompt,
                ref_image=previous_image
            )

            # Save image
            save_path = os.path.join(output_dir, f"page_{page_num+1}.png")
            image.save(save_path)

            # Update context
            self.context.add_page(text=current_text)

            # Update previous image
            previous_image = image

        print("\n🎉 All images generated successfully with continuity!")


# ----------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------
def main():
    pdf_path = input("Enter Book PDF Path: ")

    reader = PDFReader(pdf_path)
    context = ContextManager(window_size=3)
    builder = PromptBuilder()
    generator = ImageGenerator(device="mps")

    pipeline = BookToImagePipeline(reader, context, builder, generator)
    pipeline.run(output_dir="./output_images")


if __name__ == "__main__":
    main()
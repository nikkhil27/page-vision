import pypdf
import torch
from diffusers import StableDiffusionPipeline
import os

class PDFReader:
    def __init__(self, pdf_path, min_text_length=200):
        self.pdf_path = pdf_path
        self.pdf = pypdf.PdfReader(pdf_path)
        self.min_text_length = min_text_length
        self.start_page = self._detect_first_real_page()

    def _detect_first_real_page(self):
        for i, page in enumerate(self.pdf.pages):
            text = page.extract_text() or ""
            if len(text.strip()) > self.min_text_length:
                return i
        return 0  # fallback

    def get_page_text(self, page_num):
        real_page = self.start_page + page_num
        if real_page >= len(self.pdf.pages):
            return ""
        return self.pdf.pages[real_page].extract_text() or ""

    def num_pages(self):
        """Return the number of pages after skipping front matter."""
        return max(0, len(self.pdf.pages) - self.start_page)

    def get_page(self, page_num):
        return self.get_page_text(page_num)

class ContextManager():
    def __init__(self,window_size:int):
        self.window_size = window_size
        self.buffer = []
    
    def add_page(self,text:str):
        text = " ".join(text.split())
        self.buffer.append(text)

        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
    
    def get_context(self) -> str:
        return " ".join(self.buffer)
        
class PromptBuilder:
    def __init__(self, style: str = "storybook illustration, soft lighting, detailed"):
        self.style = style

    def clean(self, text: str) -> str:
        # Remove weird whitespace, line breaks, tabs, multiple spaces
        return " ".join(text.split())

    def build_prompt(self, current: str, context: str) -> str:
        current = self.clean(current)
        context = self.clean(context)

        prompt = f"""
        An illustration inspired by the following story.

        Previous context:
        {context}

        Current page:
        {current}

        Style: {self.style}.
        Highly detailed, consistent characters, coherent scene representation.
        """
        return " ".join(prompt.split())


class ImageGenerator():
    def __init__(self, device:str = "mps"):
        print("Loading Stable Diffusion Pipeline...")

        self.device = device
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/sd-turbo",
            dtype = torch.float16,
            variant = "fp16").to(device)
    
    def generate_image(self,prompt:str,steps:int = 30,guidance:float = 7.5):
        image = self.pipe(
            prompt = prompt,
            num_inference_steps = steps,
            guidance_scale = guidance
        ).images[0]
        return image

class BooktoImagePipeline():
    def __init__(self, pdfreader,context_manager,prompt_builder,generator):
        self.reader = pdfreader
        self.context_manager = context_manager
        self.prompt_builder = prompt_builder
        self.generator = generator

    def run(self, output_dir:str):

        os.makedirs(output_dir, exist_ok=True)

        total_pages = self.reader.num_pages()
        start_page = 0
        end_page = total_pages
        if start_page < 0 or end_page > total_pages:
            raise ValueError("Invalid page number")

        print(f"\n Total Pages: {total_pages} ")
        print(f"Starting from page {start_page}")

        for page_number in range(start_page, end_page):
            print(f"➡️ Processing page {page_number + 1}/{total_pages}")
            
            current_text = self.reader.get_page(page_number)

            self.context_manager.add_page(current_text)
            context = self.context_manager.get_context()

            prompt = self.prompt_builder.build_prompt(
                current = current_text,
                context = context
            )

            image = self.generator.generate_image(prompt)

            output_path = os.path.join(output_dir, f"page_{page_number+1}.png")
            image.save(output_path)

        print("🎉 All images generated successfully!")

def main():
    pdf_path = input("Enter Book PDF Path: ")

    reader = PDFReader(pdf_path)
    context_manager = ContextManager(window_size=3)
    prompt_builder = PromptBuilder(style="storybook illustration")
    generator = ImageGenerator(device="mps")

    pipeline = BooktoImagePipeline(reader, context_manager, prompt_builder, generator)
    pipeline.run(output_dir="./output_images")


if __name__ == "__main__":
    main()

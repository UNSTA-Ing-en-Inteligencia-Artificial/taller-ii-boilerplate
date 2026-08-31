import requests
from bs4 import BeautifulSoup

URL_A_EXTRAER = "https://web-scraping.dev/api/testimonials"

def descargar_html(url, pagina):
    try:
        respuesta = requests.get(f"{url}?page={pagina}", headers={"referer": "https://web-scraping.dev/testimonials"})
        if respuesta.status_code == 200:
            return respuesta.text
        else:
            return None
    except requests.exceptions.ConnectionError:
        return None

def parsear_html(html_crudo):
    parser = BeautifulSoup(html_crudo, 'html.parser')
    return parser


html_crudo = descargar_html(URL_A_EXTRAER, 2)
parser = parsear_html(html_crudo)

contenidos = parser.find_all("div", {"class": "testimonial"})
for contenido in contenidos:
    print(contenido.text.strip())

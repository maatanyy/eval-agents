from google import genai

client = genai.Client(api_key="AIzaSyB_hTciWlPob0ML3p4xs_T0A-5kHmBmjj4")

response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents="안녕",
)

print(response)
using Microsoft.AspNetCore.Mvc;
using System.Net.Http;       
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using CarPriceWebApi.Models;

namespace CarPriceWebApi.Controllers
{
    [ApiController]
    [Route("[controller]")]
    public class CarPriceController : ControllerBase
    {
        private readonly HttpClient _httpClient;

        public CarPriceController(IHttpClientFactory httpClientFactory)
        {
            _httpClient = httpClientFactory.CreateClient();
        }

        public class PredictionResponse
        {
            public float Predicted_Price { get; set; }
        }

        [HttpPost("predict/{model_name}")]
        public async Task<IActionResult> Predict([FromBody] Car features, [FromRoute] string model_name)
        {
            var options = new JsonSerializerOptions
            {
                PropertyNamingPolicy = null 
            };
            var url = $"http://localhost:8000/predict/{model_name}";
            var response = await _httpClient.PostAsJsonAsync(url, features, options);

            if (response.IsSuccessStatusCode)
            {
                var prediction = await response.Content.ReadFromJsonAsync<PredictionResponse>();
                return Ok(prediction);
            }
            return StatusCode((int)response.StatusCode, "Error has occured while predicting...");
        }

        [HttpPost("evaluate/{model_name}")]
        public async Task<IActionResult> Evaluate([FromRoute] string model_name){

            var url = $"http://localhost:8000/evaluate/{model_name}";
            var response = await _httpClient.PostAsync(url,null);

            if (response.IsSuccessStatusCode)
            {
                var options = new JsonSerializerOptions
                {
                    PropertyNamingPolicy = JsonNamingPolicy.CamelCase
                };

                var result = await response.Content.ReadFromJsonAsync<EvaluationResult>(options);
                return Ok(result);
            }
            return StatusCode((int)response.StatusCode, "Error has occured while evaluating...");
        }
    }
    
}
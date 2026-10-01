from pydantic import BaseModel
import httpx


class GeocodingResult(BaseModel):
    name: str
    latitude: float
    longitude: float
    country: str | None = None
    admin1: str | None = None


class GeocodingError(RuntimeError):
    pass


async def geocode_location(location: str, client: httpx.AsyncClient | None = None) -> GeocodingResult:
    params = {"name": location, "count": 1, "language": "en", "format": "json"}
    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=10)
    try:
        response = await client.get("https://geocoding-api.open-meteo.com/v1/search", params=params)
        response.raise_for_status()
        results = response.json().get("results", [])
        if not results:
            raise GeocodingError(f"No location found for '{location}'.")
        return GeocodingResult(**results[0])
    except GeocodingError:
        raise
    except (httpx.HTTPError, ValueError) as exc:
        raise GeocodingError("Location service could not be reached.") from exc
    finally:
        if owns_client:
            await client.aclose()

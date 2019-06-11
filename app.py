#!/usr/bin/env python
import asyncio
from aiohttp import web, web_request, ClientSession, TCPConnector
from ocr import parse_image, ParseError


routes = web.RouteTableDef()


@routes.post('/parse')
async def parse(request: web_request.Request):
    if not request.can_read_body:
        raise web.HTTPBadRequest(text='Body is empty')

    data = await request.json()

    if 'url' not in data:
        raise web.HTTPBadRequest(text='"url" is required')

    photo = await download_photo(data['url'])

    if photo is None:
        raise web.HTTPBadRequest(text='Error, cannot load image by url')

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, parse_image, photo)

        if result is None:
            raise web.HTTPBadRequest(text='NIK not found')

        return web.json_response(result)
    except ParseError as e:
        raise web.HTTPBadRequest(text=str(e))


async def download_photo(url):
    try:
        async with ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status is not 200:
                    return None
                return await resp.read()
    except Exception:
        return None


app = web.Application()
app.add_routes(routes)
web.run_app(app, host='0.0.0.0', port=8080)

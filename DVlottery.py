def draw_guidelines(img, head_info):
    """Draw improved, accurate guidelines using final canvas head_info."""
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Extract head position info (already in canvas coords)
    top_y = int(head_info.get('top_y', int(h * 0.25)))
    chin_y = int(head_info.get('chin_y', int(h * 0.75)))
    eye_y = int(head_info.get('eye_y', int(h * 0.5)))
    head_height = max(1, int(head_info.get('head_height', chin_y - top_y)))
    canvas_size = int(head_info.get('canvas_size', h))

    # Percent metrics
    actual_head_ratio = head_height / canvas_size
    actual_eye_ratio = (canvas_size - eye_y) / canvas_size

    # Outer border
    draw.rectangle([(0, 0), (w-1, h-1)], outline="gray", width=2)

    # HEAD bracket (left)
    bracket_x = 28
    draw.line([(bracket_x, top_y), (bracket_x, chin_y)], fill="blue", width=4)
    draw.line([(bracket_x-8, top_y), (bracket_x+8, top_y)], fill="blue", width=2)
    draw.line([(bracket_x-8, chin_y), (bracket_x+8, chin_y)], fill="blue", width=2)
    draw.text((bracket_x+12, max(2, top_y-14)), "Head Top", fill="blue")
    draw.text((bracket_x+12, max(2, chin_y-14)), "Chin", fill="blue")
    head_status_color = "green" if HEAD_MIN_RATIO <= actual_head_ratio <= HEAD_MAX_RATIO else "red"
    draw.text((bracket_x-120, (top_y + chin_y)//2 - 16), f"Head: {int(actual_head_ratio*100)}%", fill=head_status_color)
    draw.text((bracket_x-120, (top_y + chin_y)//2 + 2), "Req: 50-69%", fill="blue")

    # EYE bracket (right)
    eye_bracket_x = w - 28
    eye_min_y = h - int(h * EYE_MAX_RATIO)  # top edge of eye allowed band
    eye_max_y = h - int(h * EYE_MIN_RATIO)  # bottom edge of eye allowed band
    draw.line([(eye_bracket_x, eye_min_y), (eye_bracket_x, eye_max_y)], fill="green", width=4)
    draw.line([(eye_bracket_x-8, eye_min_y), (eye_bracket_x+8, eye_min_y)], fill="green", width=2)
    draw.line([(eye_bracket_x-8, eye_max_y), (eye_bracket_x+8, eye_max_y)], fill="green", width=2)
    draw.line([(eye_bracket_x-14, eye_y), (eye_bracket_x+14, eye_y)], fill="darkgreen", width=3)
    eye_status_color = "green" if EYE_MIN_RATIO <= actual_eye_ratio <= EYE_MAX_RATIO else "red"
    draw.text((eye_bracket_x-110, (eye_min_y + eye_max_y)//2 - 16), f"Eyes: {int(actual_eye_ratio*100)}%", fill=eye_status_color)
    draw.text((eye_bracket_x-110, (eye_min_y + eye_max_y)//2 + 2), "Req: 56-69%", fill="green")

    # Reference text
    draw.text((10, 10), "DV Lottery Photo Template", fill="black")
    draw.text((10, 28), f"Size: {w}x{h} px", fill="black")

    return img, actual_head_ratio, actual_eye_ratio

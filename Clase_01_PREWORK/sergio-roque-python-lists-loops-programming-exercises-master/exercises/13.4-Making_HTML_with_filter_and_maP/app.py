all_colors = [
	{"label": 'Red', "sexy": True},
	{"label": 'Pink', "sexy": False},
	{"label": 'Orange', "sexy": True},
	{"label": 'Brown', "sexy": False},
	{"label": 'Pink', "sexy": True},
	{"label": 'Violet', "sexy": True},
	{"label": 'Purple', "sexy": False},
]

# Your code here
filter_colors = list(filter(lambda x: x['sexy'],all_colors))
generate_li = list(map(lambda x: f"<li>{x['label']}</li>",filter_colors))

print(generate_li)

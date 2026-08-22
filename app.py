import os
import streamlit as st
from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM
from langchain_openai import ChatOpenAI # Import ChatOpenAI for Streamlit app

st.set_page_config(page_title="Recipe Crew", page_icon="🍳", layout="centered")
st.title("🍳 Recipe Crew")
st.caption("Three AI agents build you a full recipe: a Chef, a Nutritionist, and a Shopper.")

try:
    api_key = st.secrets["OPENAI_API_KEY"]
except (FileNotFoundError, KeyError):
    api_key = os.environ.get("OPENAI_API_KEY")

if not api_key:
    st.error("No API key found. Add OPENAI_API_KEY in Settings -> Secrets.")
    st.stop()
os.environ["OPENAI_API_KEY"] = api_key

st.sidebar.header("Settings")
diet = st.sidebar.selectbox("Dietary style", ["No restriction", "Vegetarian", "Vegan", "High-protein", "Low-carb"])
servings = st.sidebar.slider("Servings", 1, 8, 2)
skill_level = st.sidebar.radio("Cooking skill level", ["Beginner", "Intermediate", "Advanced"])

MAX_RUNS = 5
if "run_count" not in st.session_state:
    st.session_state.run_count = 0
remaining = MAX_RUNS - st.session_state.run_count
st.sidebar.metric("Runs left this session", remaining)

@st.cache_data(show_spinner=False)
def run_recipe_crew(dish_request, diet, servings, skill_level):
    openai_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7) # Instantiate ChatOpenAI
    llm = LLM(llm=openai_llm) # Wrap with CrewAI's LLM

    chef = Agent(role="Creative Chef",
        goal=f"Design one complete dish for a {skill_level.lower()} cook based on '{dish_request}', respecting a {diet} diet.",
        backstory="You trained in a busy neighborhood kitchen. You care about flavor first but keep steps realistic.",
        llm=llm, verbose=False, allow_delegation=False)

    nutritionist = Agent(role="Registered Nutritionist",
        goal="Review the Chef's dish and add a general nutrition breakdown and swaps.",
        backstory=f"You explain food without lecturing. You check the dish fits a {diet.lower()} diet.",
        llm=llm, verbose=False, allow_delegation=False)

    shopper = Agent(role="Efficient Grocery Shopper",
        goal=f"Convert the recipe into one shopping list scaled for {servings} servings, grouped by aisle.",
        backstory="You hate wasted trips, so you group items by aisle and round to real store quantities.",
        llm=llm, verbose=False, allow_delegation=False)

    critic = Agent(role="Head Judge",
        goal="Independently review the recipe, nutrition notes, and shopping list as a package.",
        backstory="You judge home-cooking competitions. You're fair but blunt, and you score out of 10.",
        llm=llm, verbose=False, allow_delegation=False)

    chef_task = Task(description=f"Create one full recipe for: '{dish_request}'. Diet: {diet}. Skill: {skill_level}. Include dish name, ingredients for {servings} servings, numbered steps.",
        expected_output="Dish name, ingredient list, numbered steps.", agent=chef)
    nutrition_task = Task(description="Using the recipe above, give an approximate nutrition breakdown and up to 3 lighter swaps. State it's general info, not medical advice.",
        expected_output="Nutrition breakdown plus optional swaps.", agent=nutritionist, context=[chef_task])
    shopping_task = Task(description=f"Using the recipe and nutrition notes above, produce one shopping list scaled for {servings} servings, grouped by section.",
        expected_output="Grouped, scaled shopping list.", agent=shopper, context=[chef_task, nutrition_task])
    critic_task = Task(description="Review the recipe, nutrition notes, and shopping list as one package. Flag mismatches. End with 'Score: X/10'.",
        expected_output="Short verdict ending in Score: X/10.", agent=critic, context=[chef_task, nutrition_task, shopping_task])

    crew = Crew(agents=[chef, nutritionist, shopper, critic],
        tasks=[chef_task, nutrition_task, shopping_task, critic_task],
        process=Process.sequential, verbose=False)
    crew.kickoff()  # no await - Streamlit has no event loop

    return {
        "recipe": str(chef_task.output),
        "nutrition": str(nutrition_task.output),
        "shopping_list": str(shopping_task.output),
        "critic": str(critic_task.output),
    }

dish_request = st.text_input("What do you feel like eating?", placeholder="e.g. something warm and spicy with chickpeas")
go = st.button("Build my recipe 🍳", type="primary", disabled=(remaining <= 0))

if remaining <= 0:
    st.warning("You've used all your runs for this session. Refresh to reset (demo limit).")

if go and dish_request.strip():
    st.session_state.run_count += 1
    with st.spinner("Chef -> Nutritionist -> Shopper -> Critic are working..."):
        output = run_recipe_crew(dish_request, diet, servings, skill_level)

    tab1, tab2, tab3, tab4 = st.tabs(["🍽️ Recipe", "📊 Nutrition", "🛒 Shopping List", "🏆 Critic"])
    with tab1: st.markdown(output["recipe"])
    with tab2: st.markdown(output["nutrition"])
    with tab3: st.markdown(output["shopping_list"])
    with tab4: st.markdown(output["critic"])

    full_text = f"RECIPE\n{output['recipe']}\n\nNUTRITION\n{output['nutrition']}\n\nSHOPPING LIST\n{output['shopping_list']}\n\nCRITIC REVIEW\n{output['critic']}\n"
    st.download_button("⬇️ Download full plan (.txt)", data=full_text, file_name="recipe_crew_plan.txt", mime="text/plain")
elif go:
    st.warning("Type what you feel like eating first.")

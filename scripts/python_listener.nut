
::FLT_MAX <- 3.402823466e+38
class :: python_listener {
    out_file_path = null;
    in_file_path = null;

    listener_bot = null;
	health = null;

	output_message = null;
	input_message = null;

	bot_angle_data =  null // bot_id -> pitch, yaw, eg. bot_angle_data[bot_id].yaw = ...
    constructor(){
		bot_angle_data = {}
		out_file_path = "squirrel_out";
    in_file_path = "squirrel_in";
		listener_bot = SpawnEntityFromTable("base_boss",
		{
			targetname = "bot",
			origin = Vector(1700,1700,0),
			model = "models/bots/skeleton_sniper/skeleton_sniper.mdl",
			playbackrate = 1.0, // Required for animations to be simulated
			// The following is done to prevent default base_boss death behavior
			// Set the health to something really big
			health = 300
		})
		// Track the health manually by using npc_hurt event and fire our custom death
		health = 300

		// Add scope to the entity
		listener_bot.ValidateScriptScope()
		local scope = listener_bot.GetScriptScope()
		// Append custom bot functionality
		scope.bot_brain <- this

		// Add behavior that will run every tick
		scope.Think <- function() {
			// Let this class handle all the work
			return bot_brain.ListenLoop()
		}
		AddThinkToEnt(listener_bot, "Think")
	}


	function Kebab() {
		printl("Jestem w Kebabie")
	}

	function DispatchAngleMessage(message) {
		// information format:
		// *bot_id* *pitch* *yaw*

		local message_parts = split(message, " ") // separating on space

		local parts_num = message_parts.len()
		if (parts_num % 3 != 1) { // well tehre is always one " " string at the front, miss by one error ig
			printl(format("Did not get good amount of parameters: %d (should be div by 3)", parts_num))
			foreach (i, part in message_parts) {
				printl( i + " " + part)
			}
			return;
		}

		bot_angle_data = null
		bot_angle_data = [] // reseting the array

		// skiping first: " "
		for (local i = 1; i < parts_num; i +=  3) {
			//removing any white characters
			message_parts[i] =  strip(message_parts[i])
			message_parts[i+1] =  strip(message_parts[i+1])
			message_parts[i+2] =  strip(message_parts[i+2])

			printl(format("1: %s, 2: %s, 3: %s.", message_parts[i], message_parts[i + 1], message_parts[i + 2]))
			local bot_id = message_parts[i].tointeger()
			local pitch = message_parts[i+1].tofloat()
			local yaw = message_parts[i+2].tofloat()


			bot_angle_data.append({
				bot_id = bot_id,
				pitch = pitch,
				yaw =  yaw
			})
		}
		foreach(id,value in bot_angle_data) {
			printl(format("1: %d, 2: %f, 3: %f.", id, value.pitch, value.yaw))
		}

		printl("Angles loaded correctly!")
		return;
	}

	function ListenLoop() {

		// Input from pyton
		input_message = FileToString(in_file_path)

		if (input_message == null) {
			return;
		}
		if (input_message == "") {
			return;
		}

		local msg = strip(input_message)
		if (msg == "exit") {
			printl("Ending program")
			StringToFile(in_file_path, "") // emptying *in* file
			listener_bot.Kill()

			FireScriptHook("Kill_BotHandler", null)
			return;
		}
		if(msg == "start") {
			printl("Starting program!")
			return;
		}
		if(msg == "\0") {
			print("Thats my own sign: \\0")
			return;
		}
		if(msg == "kebab") { // debug command
			printl("kebab")
			FireScriptHook("Set_Angles", {
				data = [
					{id=2, x=50.0, y=50.0},
					{id=3, x=67.0, y=76.0},
				]
			})
			FireScriptHook("Change_Pos", null)
		}

		StringToFile(in_file_path, "") // emptying *in* file

		DispatchAngleMessage(input_message)

		//printl("Writing to python: " + input_message)
		// Output to python
		StringToFile(out_file_path, input_message)
	}
}

p_listener <- python_listener()

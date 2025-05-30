::FLT_MAX <- 3.402823466e+38

const debug = false

class :: python_listener {
    out_file_path = null;
    in_file_path = null;

    listener_bot = null;
	health = null;

	output_message = null;
	input_message = null;

	bot_angle_data =  null

	start_program = null
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

		start_program = false
	}

	function DispatchAngleMessage(message) {
		// information format:
		// *bot_id* *pitch* *yaw*

		local message_parts = split(message, " ") // separating on space

		local parts_num = message_parts.len()
		if (parts_num % 3 != 1) { // well there is always one " " string at the front, miss by one error ig
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

			if (debug) {printl(format("1: %s, 2: %s, 3: %s.", message_parts[i], message_parts[i + 1], message_parts[i + 2]))}
			local bot_id = message_parts[i].tointeger()
			local pitch = message_parts[i+1].tofloat()
			local yaw = message_parts[i+2].tofloat()

			bot_angle_data.append({
				id = bot_id,
				y = pitch,
				x =  yaw
			})
		}

		return;
	}

	function ListenLoop() {

		// Input from pyton
		input_message = FileToString(in_file_path)
		StringToFile(in_file_path, "") // emptying *in* file

		local parts = split(input_message,  "|")

		if (input_message == null || input_message ==  "") {
			return;
		}
		local message_type = strip(parts[0])
		if (debug) {printl("input message" + input_message)}


		if (message_type == "exit") {
			printl("Ending program")
			listener_bot.Kill()
			FireScriptHook("Kill_BotHandler", null)
			FireScriptHook("KillTargetBot", null)
			return;
		}

		if(message_type == "start") {
			printl("Starting program!")
			start_program = true
		}

		if (!start_program) {
			return;
		}

		printl(message_type)
		if (message_type == "get_position") {
			if (debug) {printl("Sending Positions")}
			printl("Sending Positions")
			if (!FireScriptHook("SendPositions", null)) {
				printl("Could not fire Hook: SendPositions()")
			}

		}

		if (message_type == "angles") {
			local data = parts[1]
			DispatchAngleMessage(data)

			printl("Setting angles")
			if (!FireScriptHook("Set_Angles", {
			        data =  bot_angle_data
			    })) {
				printl("Could not fire Hook: SendPositions()")
			}

		}
		if (message_type ==  "send_damage") {
			printl("sending damage")
			if (!FireScriptHook("SendDamage", null)) {
				printl("Could not fire Hook: SendDamage()")
			}
		}

		if (message_type ==  "send_distances") {
			printl("sending distances")
			if (!FireScriptHook("SendProjectileInfo", null)) {
				printl("Could not fire Hook: SendProjectileInfo()")
			}
		}

		if (message_type == "change_shooter_pos" ) {
			if (!FireScriptHook("Change_Pos", null)) {
				printl("Could not fire Hook: Change_Pos()")
			}
		}

		if (message_type ==  "change_target_pos") {
			if (!FireScriptHook("Reposition", null)) {
				printl("Could not fire Hook: Reposition()")
			}
		}
	}
}

p_listener <- python_listener()